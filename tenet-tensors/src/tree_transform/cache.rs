use core::ops::{Add, Mul};
use std::collections::VecDeque;

use rustc_hash::FxHashMap;
use std::fs;
use std::hash::Hash;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use num_traits::Zero;
#[cfg(test)]
use tenet_core::BlockKey;
use tenet_core::{
    BlockStructure, FusionTreeBlockGroup, FusionTreeGroupKey, FusionTreeKey, FusionTreePairKey,
    GenericBraidScalar, GenericRigidSymbols, LocallyValidatedFusionTreeBlockStructure,
    MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols, SectorId, TensorMap,
    TensorStorage,
};

use crate::cache::{
    rebuild_lru_order_from_keys, touch_lru_key, typed_global_map, OperationCachePolicy,
    TreeTransformStructureCacheKey,
};
use crate::{OperationError, TreeTransformStructure, TreeTransformStructureCache};

use super::helpers::{
    duplicate_fusion_tree_pair_index, fusion_tree_group_block_keys, fusion_tree_pair_matches_group,
};
use super::operation::{
    TreeTransformBuiltinRuleCacheKey, TreeTransformOperation, TreeTransformRuleCacheKey,
};
use super::plan::{
    build_all_codomain_tree_transform_group_plan_validated,
    build_generic_tree_pair_transform_group_plan_validated,
    build_tree_pair_transform_group_plan_validated, validate_all_codomain_namespace_before_cache,
    validate_generic_tree_pair_preflight,
    validate_multiplicity_free_all_codomain_preflight_after_capability,
    validate_multiplicity_free_tree_pair_preflight,
    validate_multiplicity_free_tree_pair_preflight_after_capability,
    validate_multiplicity_free_tree_transform_capability,
    validate_tree_pair_namespace_before_cache, LocallyValidatedAllCodomainFusionTreeBlockStructure,
    TreeTransformGroupBlockSpec, TreeTransformGroupPlan,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum TreeTransformPlanScope {
    AllCodomain,
    TreePair,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct TreeTransformSectorPlanKey<RuleKey> {
    rule: RuleKey,
    scope: TreeTransformPlanScope,
    operation: TreeTransformOperation,
    source_groups: Vec<TreeTransformSourceGroupKey>,
}

impl<RuleKey> TreeTransformSectorPlanKey<RuleKey>
where
    RuleKey: Clone + Eq + Hash,
{
    pub fn tree_pair<R>(
        rule: &R,
        operation: TreeTransformOperation,
        src_structure: &BlockStructure,
    ) -> Result<Self, OperationError>
    where
        R: TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        Self::new(
            rule.tree_transform_rule_cache_key(),
            TreeTransformPlanScope::TreePair,
            operation,
            src_structure,
        )
    }

    pub fn all_codomain<R>(
        rule: &R,
        operation: TreeTransformOperation,
        src_structure: &BlockStructure,
    ) -> Result<Self, OperationError>
    where
        R: TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        Self::new(
            rule.tree_transform_rule_cache_key(),
            TreeTransformPlanScope::AllCodomain,
            operation,
            src_structure,
        )
    }

    fn new(
        rule: RuleKey,
        scope: TreeTransformPlanScope,
        operation: TreeTransformOperation,
        src_structure: &BlockStructure,
    ) -> Result<Self, OperationError> {
        let source_groups = src_structure
            .fusion_tree_groups()
            .into_iter()
            .map(|group| TreeTransformSourceGroupKey::from_group(src_structure, &group))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            rule,
            scope,
            operation,
            source_groups,
        })
    }

    #[inline]
    pub fn rule(&self) -> &RuleKey {
        &self.rule
    }

    #[inline]
    pub fn scope(&self) -> TreeTransformPlanScope {
        self.scope
    }

    #[inline]
    pub fn operation(&self) -> &TreeTransformOperation {
        &self.operation
    }

    #[inline]
    pub fn source_groups(&self) -> &[TreeTransformSourceGroupKey] {
        &self.source_groups
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct TreeTransformSourceGroupKey {
    group_key: FusionTreeGroupKey,
    src_keys: Vec<FusionTreePairKey>,
}

impl TreeTransformSourceGroupKey {
    fn from_group(
        structure: &BlockStructure,
        group: &FusionTreeBlockGroup,
    ) -> Result<Self, OperationError> {
        let src_keys = fusion_tree_group_block_keys(structure, group, "src")?;
        let Some(first) = src_keys.first() else {
            return Err(OperationError::EmptyTransformBlock);
        };
        if let Some(index) = duplicate_fusion_tree_pair_index(&src_keys) {
            return Err(OperationError::DuplicateTreeTransformKey {
                tensor: "src",
                index,
            });
        }
        let group_key = first.group_key();
        Ok(Self {
            group_key,
            src_keys,
        })
    }

    #[inline]
    pub fn group_key(&self) -> &FusionTreeGroupKey {
        &self.group_key
    }

    #[inline]
    pub fn src_keys(&self) -> &[FusionTreePairKey] {
        &self.src_keys
    }
}

#[cfg(test)]
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct TreeTransformGroupPlanKey {
    operation: TreeTransformOperation,
    groups: Vec<TreeTransformCachedGroupKey>,
}

#[cfg(test)]
impl TreeTransformGroupPlanKey {
    pub fn new<Groups>(operation: TreeTransformOperation, groups: Groups) -> Self
    where
        Groups: IntoIterator<Item = TreeTransformCachedGroupKey>,
    {
        Self {
            operation,
            groups: groups.into_iter().collect(),
        }
    }

    pub fn from_plan<T>(
        operation: TreeTransformOperation,
        plan: &TreeTransformGroupPlan<T>,
    ) -> Self {
        Self::new(
            operation,
            plan.specs()
                .iter()
                .map(TreeTransformCachedGroupKey::from_spec),
        )
    }
}

#[cfg(test)]
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct TreeTransformCachedGroupKey {
    group_key: FusionTreeGroupKey,
    dst_keys: Vec<FusionTreePairKey>,
    src_keys: Vec<FusionTreePairKey>,
}

#[cfg(test)]
impl TreeTransformCachedGroupKey {
    pub fn from_spec<T>(spec: &TreeTransformGroupBlockSpec<T>) -> Self {
        Self {
            group_key: spec.group_key().clone(),
            dst_keys: spec.dst_keys().to_vec(),
            src_keys: spec.src_keys().to_vec(),
        }
    }
}

#[cfg(test)]
#[derive(Clone, Debug)]
pub struct TreeTransformGroupPlanCache<T> {
    plans: FxHashMap<TreeTransformGroupPlanKey, TreeTransformGroupPlan<T>>,
}

#[cfg(test)]
impl<T> Default for TreeTransformGroupPlanCache<T> {
    fn default() -> Self {
        Self {
            plans: FxHashMap::default(),
        }
    }
}

#[cfg(test)]
impl<T> TreeTransformGroupPlanCache<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.plans.len()
    }

    pub fn get(&self, key: &TreeTransformGroupPlanKey) -> Option<&TreeTransformGroupPlan<T>> {
        self.plans.get(key)
    }

    pub fn insert(
        &mut self,
        key: TreeTransformGroupPlanKey,
        plan: TreeTransformGroupPlan<T>,
    ) -> Option<TreeTransformGroupPlan<T>> {
        self.plans.insert(key, plan)
    }
}

#[derive(Clone, Debug)]
pub struct TreeTransformCache<T, RuleKey> {
    plans: FxHashMap<TreeTransformSectorPlanKey<RuleKey>, Arc<TreeTransformGroupPlan<T>>>,
    plan_lru_order: VecDeque<TreeTransformSectorPlanKey<RuleKey>>,
    structures: TreeTransformStructureCache<T, TreeTransformSectorPlanKey<RuleKey>>,
    last_structure: Option<TreeTransformLastStructure<T, RuleKey>>,
    fast_structures:
        FxHashMap<TreeTransformFastStructureKey<RuleKey>, Arc<TreeTransformStructure<T>>>,
    policy: OperationCachePolicy,
    stats: TreeTransformCacheStats,
    // Why not store recoupling rows inside each plan: rows are independent of
    // degeneracy shape and remain reusable across plan-key misses.
    tree_rows: crate::tree_transform::plan::TreePairRowMemo<T, RuleKey>,
    // Why not key all-codomain rows by a full tree pair: this scope leaves the
    // domain unchanged, so the codomain tree contains every deciding input.
    all_codomain_rows: crate::tree_transform::plan::AllCodomainRowMemo<T, RuleKey>,
    // Why not expose a second compile knob: the execution context propagates
    // the backend's `recoupling_threads` to whole-group transform + assembly,
    // keeping one setting for replay and compile.
    recoupling_threads: usize,
}

type GlobalTreeTransformPlanMap<T, RuleKey> =
    RwLock<FxHashMap<TreeTransformSectorPlanKey<RuleKey>, Arc<TreeTransformGroupPlan<T>>>>;
type GlobalTreeTransformStructureMap<T, RuleKey> = RwLock<
    FxHashMap<
        TreeTransformStructureCacheKey<TreeTransformSectorPlanKey<RuleKey>>,
        Arc<TreeTransformStructure<T>>,
    >,
>;
type GlobalTreePairRowMemo<T, RuleKey> =
    RwLock<crate::tree_transform::plan::TreePairRowMemo<T, RuleKey>>;
type GlobalAllCodomainRowMemo<T, RuleKey> =
    RwLock<crate::tree_transform::plan::AllCodomainRowMemo<T, RuleKey>>;

fn global_tree_transform_plans<T, RuleKey>() -> Arc<GlobalTreeTransformPlanMap<T, RuleKey>>
where
    T: 'static + Send + Sync,
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
{
    typed_global_map()
}

fn global_tree_transform_structures<T, RuleKey>() -> Arc<GlobalTreeTransformStructureMap<T, RuleKey>>
where
    T: 'static + Send + Sync,
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
{
    typed_global_map()
}

fn global_tree_pair_rows<T, RuleKey>() -> Arc<GlobalTreePairRowMemo<T, RuleKey>>
where
    T: 'static + Send + Sync,
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
{
    typed_global_map()
}

fn global_all_codomain_rows<T, RuleKey>() -> Arc<GlobalAllCodomainRowMemo<T, RuleKey>>
where
    T: 'static + Send + Sync,
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
{
    typed_global_map()
}

#[cfg(test)]
pub(crate) fn global_tree_transform_cache_lengths<T, RuleKey>() -> (usize, usize, usize, usize)
where
    T: 'static + Send + Sync,
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
{
    let plan_len = global_tree_transform_plans::<T, RuleKey>()
        .read()
        .expect("global tree-transform plan cache poisoned")
        .len();
    let structure_len = global_tree_transform_structures::<T, RuleKey>()
        .read()
        .expect("global tree-transform structure cache poisoned")
        .len();
    let tree_row_len = global_tree_pair_rows::<T, RuleKey>()
        .read()
        .expect("global tree-pair row cache poisoned")
        .len();
    let all_codomain_row_len = global_all_codomain_rows::<T, RuleKey>()
        .read()
        .expect("global all-codomain row cache poisoned")
        .len();
    (plan_len, structure_len, tree_row_len, all_codomain_row_len)
}

const TREE_PLAN_CACHE_MAGIC: &[u8] = b"TENET_TREE_PLAN_CACHE";
const TREE_PLAN_CACHE_VERSION: u64 = 2;
const TREE_PLAN_CACHE_FILE: &str = "tree_transform_plans_v2.bin";
const LEGACY_TREE_PLAN_CACHE_FILE: &str = "tree_transform_plans_v1.bin";

fn persistent_tree_plan_cache_path() -> Option<PathBuf> {
    let dir = std::env::var_os("TENET_OPERATION_CACHE_DIR")?;
    if dir.is_empty() {
        return None;
    }
    Some(PathBuf::from(dir).join(TREE_PLAN_CACHE_FILE))
}

fn legacy_persistent_tree_plan_cache_path() -> Option<PathBuf> {
    let dir = std::env::var_os("TENET_OPERATION_CACHE_DIR")?;
    if dir.is_empty() {
        return None;
    }
    Some(PathBuf::from(dir).join(LEGACY_TREE_PLAN_CACHE_FILE))
}

fn persistent_tree_plan_loaded() -> &'static RwLock<bool> {
    static LOADED: std::sync::OnceLock<RwLock<bool>> = std::sync::OnceLock::new();
    LOADED.get_or_init(|| RwLock::new(false))
}

pub(crate) fn reset_tree_transform_persistent_cache_state() {
    *persistent_tree_plan_loaded()
        .write()
        .expect("persistent tree-plan cache state poisoned") = false;
    last_persisted_plan_count().store(0, std::sync::atomic::Ordering::Relaxed);
    if let Some(path) = persistent_tree_plan_cache_path() {
        let _ = fs::remove_file(path);
    }
    if let Some(path) = legacy_persistent_tree_plan_cache_path() {
        let _ = fs::remove_file(path);
    }
}

fn load_persistent_builtin_tree_plans_if_needed() {
    if persistent_tree_plan_cache_path().is_none() {
        return;
    }
    if *persistent_tree_plan_loaded()
        .read()
        .expect("persistent tree-plan cache state poisoned")
    {
        return;
    }
    let mut loaded = persistent_tree_plan_loaded()
        .write()
        .expect("persistent tree-plan cache state poisoned");
    if *loaded {
        return;
    }
    if let Some(path) = legacy_persistent_tree_plan_cache_path() {
        let _ = fs::remove_file(path);
    }
    let Some(path) = persistent_tree_plan_cache_path() else {
        *loaded = true;
        return;
    };
    let Ok(bytes) = fs::read(path) else {
        *loaded = true;
        return;
    };
    let Ok(entries) = decode_builtin_tree_plan_cache(&bytes) else {
        *loaded = true;
        return;
    };
    let global = global_tree_transform_plans::<f64, TreeTransformBuiltinRuleCacheKey>();
    let mut plans = global
        .write()
        .expect("global tree-transform plan cache poisoned");
    for (key, plan) in entries {
        if !builtin_tree_plan_is_multiplicity_free(&key, &plan) {
            continue;
        }
        plans.entry(key).or_insert_with(|| Arc::new(plan));
    }
    *loaded = true;
}

/// Number of plans in the cache the last time it was serialized to disk.
/// Used to skip re-encoding the entire cache on every single plan miss.
fn last_persisted_plan_count() -> &'static std::sync::atomic::AtomicUsize {
    static COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    &COUNT
}

fn persist_builtin_tree_plans_if_enabled() {
    use std::sync::atomic::Ordering;
    let Some(path) = persistent_tree_plan_cache_path() else {
        return;
    };
    let global = global_tree_transform_plans::<f64, TreeTransformBuiltinRuleCacheKey>();
    let plans = global
        .read()
        .expect("global tree-transform plan cache poisoned");
    let count = plans.len();
    if count == 0 {
        return;
    }
    // ponytail: this runs on every plan miss; re-encoding the whole cache each
    // time is O(misses × cachesize) (measured: 20 GiB churn on a cross-χ run).
    // Only re-serialize once the cache has grown by a chunk. Worst case the disk
    // file lags the in-memory cache by up to last/8 plans, which the next process
    // simply recompiles — the persistent cache is an optimization, not correctness.
    let last = last_persisted_plan_count().load(Ordering::Relaxed);
    if count < last + core::cmp::max(64, last / 8) {
        return;
    }
    let Ok(bytes) = encode_builtin_tree_plan_cache(&plans) else {
        return;
    };
    last_persisted_plan_count().store(count, Ordering::Relaxed);
    if let Some(parent) = path.parent() {
        if fs::create_dir_all(parent).is_err() {
            return;
        }
    }
    let tmp = path.with_extension("bin.tmp");
    if fs::write(&tmp, bytes).is_ok() {
        let _ = fs::rename(tmp, path);
    }
}

fn encode_builtin_tree_plan_cache(
    plans: &FxHashMap<
        TreeTransformSectorPlanKey<TreeTransformBuiltinRuleCacheKey>,
        Arc<TreeTransformGroupPlan<f64>>,
    >,
) -> Result<Vec<u8>, ()> {
    let mut out = Vec::new();
    out.extend_from_slice(TREE_PLAN_CACHE_MAGIC);
    write_u64(&mut out, TREE_PLAN_CACHE_VERSION);
    // Why-not add the missing multiplicity identity bit to v2 here: the
    // imminent Opaque/FusionTree + MultiplicityIndex wire migration owns v3.
    // Until then Generic plans stay supported in memory but do not cross this
    // lossy persistent boundary, avoiding two incompatible cache bumps.
    let persistent_plans = plans
        .iter()
        .filter(|(key, plan)| builtin_tree_plan_is_multiplicity_free(key, plan));
    write_usize(&mut out, persistent_plans.clone().count());
    for (key, plan) in persistent_plans {
        encode_builtin_tree_plan_key(&mut out, key)?;
        encode_tree_transform_group_plan_f64(&mut out, plan)?;
    }
    Ok(out)
}

fn builtin_tree_plan_is_multiplicity_free(
    key: &TreeTransformSectorPlanKey<TreeTransformBuiltinRuleCacheKey>,
    plan: &TreeTransformGroupPlan<f64>,
) -> bool {
    key.source_groups()
        .iter()
        .flat_map(TreeTransformSourceGroupKey::src_keys)
        .chain(
            plan.specs()
                .iter()
                .flat_map(|spec| spec.src_keys().iter().chain(spec.dst_keys())),
        )
        .all(fusion_tree_pair_is_multiplicity_free)
}

fn fusion_tree_pair_is_multiplicity_free(tree: &FusionTreePairKey) -> bool {
    !tree.codomain_tree().has_multiplicity() && !tree.domain_tree().has_multiplicity()
}

fn decode_builtin_tree_plan_cache(
    bytes: &[u8],
) -> Result<
    Vec<(
        TreeTransformSectorPlanKey<TreeTransformBuiltinRuleCacheKey>,
        TreeTransformGroupPlan<f64>,
    )>,
    (),
> {
    let mut input = CacheBytes::new(bytes);
    input.expect_bytes(TREE_PLAN_CACHE_MAGIC)?;
    if input.read_u64()? != TREE_PLAN_CACHE_VERSION {
        return Err(());
    }
    let len = input.read_usize()?;
    let mut entries = Vec::with_capacity(len);
    for _ in 0..len {
        let key = decode_builtin_tree_plan_key(&mut input)?;
        let plan = decode_tree_transform_group_plan_f64(&mut input)?;
        if let (Some(key), Some(plan)) = (key, plan) {
            entries.push((key, plan));
        }
    }
    input.finish()?;
    Ok(entries)
}

fn encode_builtin_tree_plan_key(
    out: &mut Vec<u8>,
    key: &TreeTransformSectorPlanKey<TreeTransformBuiltinRuleCacheKey>,
) -> Result<(), ()> {
    encode_builtin_rule_key(out, *key.rule());
    encode_plan_scope(out, key.scope());
    encode_tree_transform_operation(out, key.operation());
    write_usize(out, key.source_groups().len());
    for group in key.source_groups() {
        encode_fusion_tree_group_key(out, group.group_key());
        write_usize(out, group.src_keys().len());
        for key in group.src_keys() {
            encode_v2_fusion_tree_pair_key(out, key);
        }
    }
    Ok(())
}

fn decode_builtin_tree_plan_key(
    input: &mut CacheBytes<'_>,
) -> Result<Option<TreeTransformSectorPlanKey<TreeTransformBuiltinRuleCacheKey>>, ()> {
    let rule = decode_builtin_rule_key(input)?;
    let scope = decode_plan_scope(input)?;
    let operation = decode_tree_transform_operation(input)?;
    let group_count = input.read_usize()?;
    let mut source_groups = Vec::with_capacity(group_count);
    let mut categorical = true;
    for _ in 0..group_count {
        let serialized_group_key = decode_fusion_tree_group_key(input)?;
        let key_count = input.read_usize()?;
        let mut src_keys = Vec::with_capacity(key_count);
        let mut group_is_categorical = true;
        for _ in 0..key_count {
            match decode_v2_fusion_tree_pair_key(input)? {
                Some(key) => src_keys.push(key),
                None => group_is_categorical = false,
            }
        }
        let derived_group_key = src_keys.first().map(FusionTreePairKey::group_key);
        let source_group = match derived_group_key {
            Some(group_key)
                if group_is_categorical
                    && group_key == serialized_group_key
                    && src_keys
                        .iter()
                        .all(|key| fusion_tree_pair_matches_group(key, &group_key))
                    && duplicate_fusion_tree_pair_index(&src_keys).is_none() =>
            {
                Some(TreeTransformSourceGroupKey {
                    group_key,
                    src_keys,
                })
            }
            _ => None,
        };
        if let Some(source_group) = source_group {
            source_groups.push(source_group);
        } else {
            categorical = false;
        }
    }
    Ok(categorical.then_some(TreeTransformSectorPlanKey {
        rule,
        scope,
        operation,
        source_groups,
    }))
}

fn encode_tree_transform_group_plan_f64(
    out: &mut Vec<u8>,
    plan: &TreeTransformGroupPlan<f64>,
) -> Result<(), ()> {
    write_usize(out, plan.specs().len());
    for spec in plan.specs() {
        encode_fusion_tree_group_key(out, spec.group_key());
        write_usize(out, spec.dst_keys().len());
        for key in spec.dst_keys() {
            encode_v2_fusion_tree_pair_key(out, key);
        }
        write_usize(out, spec.src_keys().len());
        for key in spec.src_keys() {
            encode_v2_fusion_tree_pair_key(out, key);
        }
        write_usize(out, spec.recoupling_coefficients_dst_src().len());
        for &coefficient in spec.recoupling_coefficients_dst_src() {
            write_u64(out, coefficient.to_bits());
        }
        match spec.source_axes() {
            Some(axes) => {
                out.push(1);
                write_usize(out, axes.len());
                for &axis in axes {
                    write_usize(out, axis);
                }
            }
            None => out.push(0),
        }
    }
    Ok(())
}

fn decode_tree_transform_group_plan_f64(
    input: &mut CacheBytes<'_>,
) -> Result<Option<TreeTransformGroupPlan<f64>>, ()> {
    let spec_count = input.read_usize()?;
    let mut specs = Vec::with_capacity(spec_count);
    let mut shared_axes = FxHashMap::<Arc<[usize]>, Arc<[usize]>>::default();
    let mut categorical = true;
    for _ in 0..spec_count {
        let serialized_group_key = decode_fusion_tree_group_key(input)?;
        let dst_count = input.read_usize()?;
        let mut dst_keys = Vec::with_capacity(dst_count);
        let mut spec_is_categorical = true;
        for _ in 0..dst_count {
            match decode_v2_fusion_tree_pair_key(input)? {
                Some(key) => dst_keys.push(key),
                None => spec_is_categorical = false,
            }
        }
        let src_count = input.read_usize()?;
        let mut src_keys = Vec::with_capacity(src_count);
        for _ in 0..src_count {
            match decode_v2_fusion_tree_pair_key(input)? {
                Some(key) => src_keys.push(key),
                None => spec_is_categorical = false,
            }
        }
        let coeff_count = input.read_usize()?;
        let mut coefficients = Vec::with_capacity(coeff_count);
        for _ in 0..coeff_count {
            coefficients.push(f64::from_bits(input.read_u64()?));
        }
        let mut spec = if spec_is_categorical {
            let candidate = if dst_count == 1 && src_count == 1 && coeff_count == 1 {
                Ok(TreeTransformGroupBlockSpec::single(
                    dst_keys.pop().ok_or(())?,
                    src_keys.pop().ok_or(())?,
                    coefficients.pop().ok_or(())?,
                ))
            } else {
                TreeTransformGroupBlockSpec::try_multi(dst_keys, src_keys, coefficients)
            };
            match candidate {
                Ok(spec) if spec.group_key() == &serialized_group_key => Some(spec),
                Ok(_) | Err(_) => {
                    categorical = false;
                    None
                }
            }
        } else {
            categorical = false;
            None
        };
        if input.read_u8()? != 0 {
            let axis_count = input.read_usize()?;
            let mut axes = Vec::with_capacity(axis_count);
            for _ in 0..axis_count {
                axes.push(input.read_usize()?);
            }
            let axes: Arc<[usize]> = axes.into();
            let axes = shared_axes.get(axes.as_ref()).cloned().unwrap_or_else(|| {
                shared_axes.insert(Arc::clone(&axes), Arc::clone(&axes));
                axes
            });
            spec = spec.map(|spec| spec.with_shared_source_axes(axes));
        }
        if let Some(spec) = spec {
            specs.push(spec);
        }
    }
    Ok(categorical.then(|| TreeTransformGroupPlan::new(specs)))
}

fn encode_builtin_rule_key(out: &mut Vec<u8>, key: TreeTransformBuiltinRuleCacheKey) {
    out.push(match key {
        TreeTransformBuiltinRuleCacheKey::Z2 => 0,
        TreeTransformBuiltinRuleCacheKey::FermionParity => 1,
        TreeTransformBuiltinRuleCacheKey::U1 => 2,
        TreeTransformBuiltinRuleCacheKey::SU2Exact { .. } => 3,
    });
    if let TreeTransformBuiltinRuleCacheKey::SU2Exact { authority_version } = key {
        out.push(authority_version);
    }
}

fn decode_builtin_rule_key(
    input: &mut CacheBytes<'_>,
) -> Result<TreeTransformBuiltinRuleCacheKey, ()> {
    match input.read_u8()? {
        0 => Ok(TreeTransformBuiltinRuleCacheKey::Z2),
        1 => Ok(TreeTransformBuiltinRuleCacheKey::FermionParity),
        2 => Ok(TreeTransformBuiltinRuleCacheKey::U1),
        3 => {
            let authority_version = input.read_u8()?;
            if authority_version != tenet_core::SU2_EXACT_AUTHORITY_VERSION {
                return Err(());
            }
            Ok(TreeTransformBuiltinRuleCacheKey::SU2Exact { authority_version })
        }
        _ => Err(()),
    }
}

fn encode_plan_scope(out: &mut Vec<u8>, scope: TreeTransformPlanScope) {
    out.push(match scope {
        TreeTransformPlanScope::AllCodomain => 0,
        TreeTransformPlanScope::TreePair => 1,
    });
}

fn decode_plan_scope(input: &mut CacheBytes<'_>) -> Result<TreeTransformPlanScope, ()> {
    match input.read_u8()? {
        0 => Ok(TreeTransformPlanScope::AllCodomain),
        1 => Ok(TreeTransformPlanScope::TreePair),
        _ => Err(()),
    }
}

fn encode_tree_transform_operation(out: &mut Vec<u8>, operation: &TreeTransformOperation) {
    match operation {
        TreeTransformOperation::Transpose {
            codomain_permutation,
            domain_permutation,
        } => {
            out.push(0);
            encode_usize_slice(out, codomain_permutation);
            encode_usize_slice(out, domain_permutation);
        }
        TreeTransformOperation::Permute {
            codomain_permutation,
            domain_permutation,
        } => {
            out.push(1);
            encode_usize_slice(out, codomain_permutation);
            encode_usize_slice(out, domain_permutation);
        }
        TreeTransformOperation::Braid {
            codomain_permutation,
            domain_permutation,
            codomain_levels,
            domain_levels,
        } => {
            out.push(2);
            encode_usize_slice(out, codomain_permutation);
            encode_usize_slice(out, domain_permutation);
            encode_usize_slice(out, codomain_levels);
            encode_usize_slice(out, domain_levels);
        }
    }
}

fn decode_tree_transform_operation(
    input: &mut CacheBytes<'_>,
) -> Result<TreeTransformOperation, ()> {
    match input.read_u8()? {
        0 => Ok(TreeTransformOperation::transpose(
            decode_usize_vec(input)?,
            decode_usize_vec(input)?,
        )),
        1 => Ok(TreeTransformOperation::permute(
            decode_usize_vec(input)?,
            decode_usize_vec(input)?,
        )),
        2 => Ok(TreeTransformOperation::braid(
            decode_usize_vec(input)?,
            decode_usize_vec(input)?,
            decode_usize_vec(input)?,
            decode_usize_vec(input)?,
        )),
        _ => Err(()),
    }
}

fn encode_fusion_tree_group_key(out: &mut Vec<u8>, key: &FusionTreeGroupKey) {
    encode_sector_slice(out, key.codomain_uncoupled());
    encode_sector_slice(out, key.domain_uncoupled());
    encode_bool_slice(out, key.codomain_is_dual());
    encode_bool_slice(out, key.domain_is_dual());
}

fn decode_fusion_tree_group_key(input: &mut CacheBytes<'_>) -> Result<FusionTreeGroupKey, ()> {
    Ok(FusionTreeGroupKey::new(
        decode_sector_vec(input)?,
        decode_sector_vec(input)?,
        decode_bool_vec(input)?,
        decode_bool_vec(input)?,
    ))
}

fn encode_v2_fusion_tree_pair_key(out: &mut Vec<u8>, tree: &FusionTreePairKey) {
    out.push(1);
    encode_fusion_tree_pair_key(out, tree);
}

fn decode_v2_fusion_tree_pair_key(
    input: &mut CacheBytes<'_>,
) -> Result<Option<FusionTreePairKey>, ()> {
    match input.read_u8()? {
        // Why not fail on the legacy dense tag: v2 files can contain later
        // categorical entries. The decoder must consume this whole entry and
        // discard it without losing the following wire position.
        0 => Ok(None),
        1 => Ok(Some(decode_fusion_tree_pair_key(input)?)),
        _ => Err(()),
    }
}

fn encode_fusion_tree_pair_key(out: &mut Vec<u8>, key: &FusionTreePairKey) {
    encode_fusion_tree_key(out, key.codomain_tree());
    encode_fusion_tree_key(out, key.domain_tree());
}

fn decode_fusion_tree_pair_key(input: &mut CacheBytes<'_>) -> Result<FusionTreePairKey, ()> {
    let codomain = decode_fusion_tree_key_parts(input)?;
    let domain = decode_fusion_tree_key_parts(input)?;
    if codomain.coupled != domain.coupled {
        return Err(());
    }
    Ok(FusionTreePairKey::pair_from_sector_ids(
        codomain.uncoupled.into_iter().map(SectorId::id),
        domain.uncoupled.into_iter().map(SectorId::id),
        codomain.coupled.map(SectorId::id),
        codomain.is_dual,
        domain.is_dual,
        codomain.innerlines.into_iter().map(SectorId::id),
        domain.innerlines.into_iter().map(SectorId::id),
        codomain.vertices.into_iter().map(SectorId::id),
        domain.vertices.into_iter().map(SectorId::id),
    ))
}

fn encode_fusion_tree_key(out: &mut Vec<u8>, key: &FusionTreeKey) {
    encode_sector_slice(out, key.uncoupled());
    match key.coupled() {
        Some(coupled) => {
            out.push(1);
            write_usize(out, coupled.id());
        }
        None => out.push(0),
    }
    encode_bool_slice(out, key.is_dual());
    encode_sector_slice(out, key.innerlines());
    encode_sector_slice(out, key.vertices());
}

struct DecodedFusionTreeKey {
    uncoupled: Vec<SectorId>,
    coupled: Option<SectorId>,
    is_dual: Vec<bool>,
    innerlines: Vec<SectorId>,
    vertices: Vec<SectorId>,
}

fn decode_fusion_tree_key_parts(input: &mut CacheBytes<'_>) -> Result<DecodedFusionTreeKey, ()> {
    let uncoupled = decode_sector_vec(input)?;
    let coupled = if input.read_u8()? == 0 {
        None
    } else {
        Some(SectorId::new(input.read_usize()?))
    };
    let is_dual = decode_bool_vec(input)?;
    let innerlines = decode_sector_vec(input)?;
    let vertices = decode_sector_vec(input)?;
    Ok(DecodedFusionTreeKey {
        uncoupled,
        coupled,
        is_dual,
        innerlines,
        vertices,
    })
}

fn encode_sector_slice(out: &mut Vec<u8>, sectors: &[SectorId]) {
    write_usize(out, sectors.len());
    for &sector in sectors {
        write_usize(out, sector.id());
    }
}

fn decode_sector_vec(input: &mut CacheBytes<'_>) -> Result<Vec<SectorId>, ()> {
    let len = input.read_usize()?;
    let mut sectors = Vec::with_capacity(len);
    for _ in 0..len {
        sectors.push(SectorId::new(input.read_usize()?));
    }
    Ok(sectors)
}

fn encode_usize_slice(out: &mut Vec<u8>, values: &[usize]) {
    write_usize(out, values.len());
    for &value in values {
        write_usize(out, value);
    }
}

fn decode_usize_vec(input: &mut CacheBytes<'_>) -> Result<Vec<usize>, ()> {
    let len = input.read_usize()?;
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        values.push(input.read_usize()?);
    }
    Ok(values)
}

fn encode_bool_slice(out: &mut Vec<u8>, values: &[bool]) {
    write_usize(out, values.len());
    for &value in values {
        out.push(u8::from(value));
    }
}

fn decode_bool_vec(input: &mut CacheBytes<'_>) -> Result<Vec<bool>, ()> {
    let len = input.read_usize()?;
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        values.push(match input.read_u8()? {
            0 => false,
            1 => true,
            _ => return Err(()),
        });
    }
    Ok(values)
}

fn write_usize(out: &mut Vec<u8>, value: usize) {
    write_u64(out, value as u64);
}

fn write_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

struct CacheBytes<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> CacheBytes<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn expect_bytes(&mut self, expected: &[u8]) -> Result<(), ()> {
        let end = self.pos.checked_add(expected.len()).ok_or(())?;
        if self.bytes.get(self.pos..end) != Some(expected) {
            return Err(());
        }
        self.pos = end;
        Ok(())
    }

    fn read_u8(&mut self) -> Result<u8, ()> {
        let value = *self.bytes.get(self.pos).ok_or(())?;
        self.pos += 1;
        Ok(value)
    }

    fn read_u64(&mut self) -> Result<u64, ()> {
        let end = self.pos.checked_add(8).ok_or(())?;
        let bytes = self.bytes.get(self.pos..end).ok_or(())?;
        self.pos = end;
        Ok(u64::from_le_bytes(bytes.try_into().map_err(|_| ())?))
    }

    fn read_usize(&mut self) -> Result<usize, ()> {
        usize::try_from(self.read_u64()?).map_err(|_| ())
    }

    fn finish(self) -> Result<(), ()> {
        if self.pos == self.bytes.len() {
            Ok(())
        } else {
            Err(())
        }
    }
}

#[cfg(test)]
mod persistence_tests {
    use super::*;
    use tenet_core::{BraidingStyleKind, FusionRule, FusionStyleKind, RuleIdentity, SectorVec};

    #[derive(Clone, Copy)]
    struct GenericCacheProbeRule;

    impl FusionRule for GenericCacheProbeRule {
        fn rule_identity(&self) -> RuleIdentity {
            RuleIdentity::of_type::<Self>()
        }

        fn fusion_style(&self) -> FusionStyleKind {
            FusionStyleKind::Generic
        }

        fn braiding_style(&self) -> BraidingStyleKind {
            BraidingStyleKind::Bosonic
        }

        fn vacuum(&self) -> SectorId {
            SectorId::new(0)
        }

        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            if left == self.vacuum() {
                [right].into_iter().collect()
            } else if right == self.vacuum() {
                [left].into_iter().collect()
            } else {
                SectorVec::new()
            }
        }
    }

    fn multiplicity_free_key() -> FusionTreePairKey {
        FusionTreePairKey::pair_from_sector_ids(
            [1, 1],
            [],
            Some(0),
            [false, false],
            [],
            [],
            [],
            [1],
            [],
        )
    }

    fn decode_hex_fixture(hex: &str) -> Vec<u8> {
        fn nibble(byte: u8) -> u8 {
            match byte {
                b'0'..=b'9' => byte - b'0',
                b'a'..=b'f' => byte - b'a' + 10,
                b'A'..=b'F' => byte - b'A' + 10,
                _ => panic!("persistent fixture contains a non-hex byte"),
            }
        }

        assert_eq!(hex.len() % 2, 0);
        hex.as_bytes()
            .chunks_exact(2)
            .map(|pair| (nibble(pair[0]) << 4) | nibble(pair[1]))
            .collect()
    }

    fn origin_main_legacy_dense_v2_fixture() -> Vec<u8> {
        // What: bytes produced by origin/main 35de652's encoder with two one-by-one
        // `multi` specs sharing source axes [0, 1]. Keeping literal bytes
        // catches representation migrations that a current-encoder roundtrip
        // cannot see.
        const ORIGIN_MAIN_35DE652_V2: &str = concat!(
            "54454e45545f545245455f504c414e5f43414348450200000000000000010000",
            "0000000000030101020200000000000000000000000000000001000000000000",
            "0000000000000000000200000000000000030000000000000005000000000000",
            "0000000000000000000100000000000000020000000000000001000000000000",
            "0001000000000000000000000000000000020000000000000000000000000000",
            "0000000100000000000000000200000000000000020000000000000001000000",
            "0000000001000000000000000000000000000000020000000000000000000000",
            "0000000000000100000000000000000100000000000000000100000000000000",
            "000000000000f43f010200000000000000000000000000000001000000000000",
            "0002000000000000000100000000000000010000000000000000000000000000",
            "0002000000000000000000000000000000000001000000000000000001000000",
            "0000000000010000000000000000000000000004c00102000000000000000000",
            "0000000000000100000000000000",
        );
        decode_hex_fixture(ORIGIN_MAIN_35DE652_V2)
    }

    fn base_87065ed_categorical_matrix_v2_fixture() -> Vec<u8> {
        // What: whole-file bytes produced by the exact 87065ed encoder for one
        // two-by-two SU(2) categorical cohort. A literal from the predecessor
        // implementation detects one-sided writer or reader drift.
        decode_hex_fixture(include_str!(
            "fixtures/base_87065ed_categorical_matrix_v2.hex"
        ))
    }

    #[test]
    fn origin_main_v2_fixture_consumes_and_drops_legacy_dense_plan() {
        let bytes = origin_main_legacy_dense_v2_fixture();
        let decoded = decode_builtin_tree_plan_cache(&bytes).unwrap();

        // What: the decoder fully consumes a valid legacy v2 record but drops
        // its whole plan once any source or destination key is Dense.
        assert!(decoded.is_empty());
    }

    #[test]
    fn base_87065ed_categorical_v2_bytes_preserve_exact_matrix_contract() {
        let pair = |inner| {
            FusionTreePairKey::pair_from_sector_ids(
                [1; 4],
                [],
                Some(0),
                [false; 4],
                [],
                inner,
                [],
                [1; 3],
                [],
            )
        };
        let first = pair([0, 1]);
        let second = pair([2, 1]);
        let key = TreeTransformSectorPlanKey {
            rule: TreeTransformBuiltinRuleCacheKey::SU2Exact {
                authority_version: tenet_core::SU2_EXACT_AUTHORITY_VERSION,
            },
            scope: TreeTransformPlanScope::TreePair,
            operation: TreeTransformOperation::braid([1, 0, 2, 3], [], [5, 3, 0, 0], []),
            source_groups: vec![TreeTransformSourceGroupKey {
                group_key: first.group_key(),
                src_keys: vec![first.clone(), second.clone()],
            }],
        };
        let plan = TreeTransformGroupPlan::new(vec![TreeTransformGroupBlockSpec::try_multi(
            [second.clone(), first.clone()],
            [first.clone(), second.clone()],
            vec![1.25, -2.5, 3.75, -4.5],
        )
        .unwrap()
        .with_source_axes([1, 0, 2, 3])]);
        let literal = base_87065ed_categorical_matrix_v2_fixture();
        let mut entries = FxHashMap::default();
        entries.insert(key.clone(), Arc::new(plan.clone()));

        // What: the current writer emits the exact whole-file categorical v2
        // representation produced by 87065ed, independently of current decode.
        let encoded = encode_builtin_tree_plan_cache(&entries).unwrap();
        assert_eq!(literal.len(), 1_348);
        assert_eq!(encoded.len(), literal.len());
        if let Some(index) = encoded
            .iter()
            .zip(&literal)
            .position(|(current, base)| current != base)
        {
            panic!(
                "categorical v2 byte differs at {index}: current={:02x}, base={:02x}",
                encoded[index], literal[index]
            );
        }
        assert_eq!(encoded, literal);

        let decoded = decode_builtin_tree_plan_cache(&literal).unwrap();
        assert_eq!(decoded, vec![(key, plan)]);
        let spec = &decoded[0].1.specs()[0];

        // What: decoding preserves destination-row/source-column order,
        // row-major U[dst, src] coefficients, and the source-axis map exactly.
        assert_eq!(spec.dst_keys(), &[second.clone(), first.clone()]);
        assert_eq!(spec.src_keys(), &[first, second]);
        assert_eq!(
            spec.recoupling_coefficients_dst_src(),
            &[1.25, -2.5, 3.75, -4.5]
        );
        assert_eq!(spec.source_axes(), Some([1, 0, 2, 3].as_slice()));
    }

    fn categorical_v2_fixture() -> (
        TreeTransformSectorPlanKey<TreeTransformBuiltinRuleCacheKey>,
        TreeTransformGroupPlan<f64>,
        Vec<u8>,
    ) {
        let tree = multiplicity_free_key();
        let group_key = tree.group_key();
        let categorical_key = TreeTransformSectorPlanKey {
            rule: TreeTransformBuiltinRuleCacheKey::SU2Exact {
                authority_version: tenet_core::SU2_EXACT_AUTHORITY_VERSION,
            },
            scope: TreeTransformPlanScope::TreePair,
            operation: TreeTransformOperation::braid([1, 0], [], [5, 3], []),
            source_groups: vec![TreeTransformSourceGroupKey {
                group_key: group_key.clone(),
                src_keys: vec![tree.clone()],
            }],
        };
        let categorical_plan =
            TreeTransformGroupPlan::new(vec![TreeTransformGroupBlockSpec::single(
                tree.clone(),
                tree,
                3.25,
            )
            .with_source_axes([1, 0])]);
        let mut categorical_entries = FxHashMap::default();
        categorical_entries.insert(categorical_key.clone(), Arc::new(categorical_plan.clone()));
        let categorical_bytes = encode_builtin_tree_plan_cache(&categorical_entries).unwrap();
        (categorical_key, categorical_plan, categorical_bytes)
    }

    #[test]
    fn v2_dense_record_does_not_misalign_following_categorical_record() {
        let (categorical_key, categorical_plan, categorical_bytes) = categorical_v2_fixture();
        let legacy_bytes = origin_main_legacy_dense_v2_fixture();
        let records_offset = TREE_PLAN_CACHE_MAGIC.len() + 8 + 8;

        assert_eq!(
            &legacy_bytes[..TREE_PLAN_CACHE_MAGIC.len() + 8],
            &categorical_bytes[..TREE_PLAN_CACHE_MAGIC.len() + 8]
        );
        let mut mixed = legacy_bytes[..TREE_PLAN_CACHE_MAGIC.len() + 8].to_vec();
        write_usize(&mut mixed, 2);
        mixed.extend_from_slice(&legacy_bytes[records_offset..]);
        mixed.extend_from_slice(&categorical_bytes[records_offset..]);

        let decoded = decode_builtin_tree_plan_cache(&mixed).unwrap();

        // What: one real v2 stream drops the complete legacy Dense record and
        // resumes at the exact boundary of the following categorical record.
        assert_eq!(decoded, vec![(categorical_key, categorical_plan)]);
        assert_eq!(
            decoded[0].1.specs()[0].recoupling_coefficients_dst_src(),
            &[3.25]
        );
        assert_eq!(
            decoded[0].1.specs()[0].source_axes(),
            Some([1, 0].as_slice())
        );
        assert!(format!("{:?}", decoded[0].1.specs()[0]).contains("Single"));
    }

    #[test]
    fn v2_mismatched_group_does_not_misalign_following_categorical_record() {
        let (categorical_key, categorical_plan, categorical_bytes) = categorical_v2_fixture();
        let records_offset = TREE_PLAN_CACHE_MAGIC.len() + 8 + 8;
        let plan_offset = {
            let mut record = CacheBytes::new(&categorical_bytes[records_offset..]);
            assert!(decode_builtin_tree_plan_key(&mut record).unwrap().is_some());
            records_offset + record.pos
        };
        let first_plan_group_sector = plan_offset + 8 + 8;
        let mut mismatched = categorical_bytes.clone();
        mismatched[first_plan_group_sector..first_plan_group_sector + 8]
            .copy_from_slice(&99_u64.to_le_bytes());
        let mut mixed = categorical_bytes[..TREE_PLAN_CACHE_MAGIC.len() + 8].to_vec();
        write_usize(&mut mixed, 2);
        mixed.extend_from_slice(&mismatched[records_offset..]);
        mixed.extend_from_slice(&categorical_bytes[records_offset..]);

        let decoded = decode_builtin_tree_plan_cache(&mixed).unwrap();

        // What: a legacy categorical record whose serialized source group
        // disagrees with its source pairs is dropped only after its full wire
        // body is consumed, so the next record remains aligned and exact.
        assert_eq!(decoded, vec![(categorical_key, categorical_plan)]);
        assert_eq!(
            decoded[0].1.specs()[0].recoupling_coefficients_dst_src(),
            &[3.25]
        );
        assert_eq!(
            decoded[0].1.specs()[0].source_axes(),
            Some([1, 0].as_slice())
        );
    }

    fn two_source_categorical_v2_fixture() -> Vec<u8> {
        let src1 = multiplicity_free_key();
        let src2 = FusionTreePairKey::pair_from_sector_ids(
            [1, 1],
            [],
            Some(0),
            [false, false],
            [],
            [],
            [],
            [2],
            [],
        );
        let group_key = src1.group_key();
        let duplicate_key = TreeTransformSectorPlanKey {
            rule: TreeTransformBuiltinRuleCacheKey::SU2Exact {
                authority_version: tenet_core::SU2_EXACT_AUTHORITY_VERSION,
            },
            scope: TreeTransformPlanScope::TreePair,
            operation: TreeTransformOperation::braid([1, 0], [], [5, 3], []),
            source_groups: vec![TreeTransformSourceGroupKey {
                group_key,
                src_keys: vec![src1.clone(), src2.clone()],
            }],
        };
        let duplicate_plan =
            TreeTransformGroupPlan::new(vec![TreeTransformGroupBlockSpec::try_multi(
                [src1.clone()],
                [src1.clone(), src2],
                vec![1.0, 2.0],
            )
            .unwrap()]);
        let mut duplicate_entries = FxHashMap::default();
        duplicate_entries.insert(duplicate_key, Arc::new(duplicate_plan));
        encode_builtin_tree_plan_cache(&duplicate_entries).unwrap()
    }

    #[test]
    fn v2_duplicate_plan_source_key_drops_only_its_record() {
        let mut duplicate_bytes = two_source_categorical_v2_fixture();
        let records_offset = TREE_PLAN_CACHE_MAGIC.len() + 8 + 8;
        let (first_src, second_src) = {
            let mut record = CacheBytes::new(&duplicate_bytes[records_offset..]);
            assert!(decode_builtin_tree_plan_key(&mut record).unwrap().is_some());
            assert_eq!(record.read_usize().unwrap(), 1);
            decode_fusion_tree_group_key(&mut record).unwrap();
            assert_eq!(record.read_usize().unwrap(), 1);
            assert!(decode_v2_fusion_tree_pair_key(&mut record)
                .unwrap()
                .is_some());
            assert_eq!(record.read_usize().unwrap(), 2);
            let first_start = records_offset + record.pos;
            assert!(decode_v2_fusion_tree_pair_key(&mut record)
                .unwrap()
                .is_some());
            let first_end = records_offset + record.pos;
            let second_start = first_end;
            assert!(decode_v2_fusion_tree_pair_key(&mut record)
                .unwrap()
                .is_some());
            let second_end = records_offset + record.pos;
            (first_start..first_end, second_start..second_end)
        };
        assert_eq!(first_src.len(), second_src.len());
        let first_src_bytes = duplicate_bytes[first_src].to_vec();
        duplicate_bytes[second_src].copy_from_slice(&first_src_bytes);
        let (categorical_key, categorical_plan, categorical_bytes) = categorical_v2_fixture();
        let mut mixed = categorical_bytes[..TREE_PLAN_CACHE_MAGIC.len() + 8].to_vec();
        write_usize(&mut mixed, 2);
        mixed.extend_from_slice(&duplicate_bytes[records_offset..]);
        mixed.extend_from_slice(&categorical_bytes[records_offset..]);

        let decoded = decode_builtin_tree_plan_cache(&mixed).unwrap();

        // What: duplicate categorical columns invalidate one legacy plan after
        // complete consumption without disturbing the following record.
        assert_eq!(decoded, vec![(categorical_key, categorical_plan)]);
    }

    #[test]
    fn v2_duplicate_cache_key_source_drops_only_its_record() {
        let mut duplicate_bytes = two_source_categorical_v2_fixture();
        let records_offset = TREE_PLAN_CACHE_MAGIC.len() + 8 + 8;
        let (first_src, second_src) = {
            let mut record = CacheBytes::new(&duplicate_bytes[records_offset..]);
            decode_builtin_rule_key(&mut record).unwrap();
            decode_plan_scope(&mut record).unwrap();
            decode_tree_transform_operation(&mut record).unwrap();
            assert_eq!(record.read_usize().unwrap(), 1);
            decode_fusion_tree_group_key(&mut record).unwrap();
            assert_eq!(record.read_usize().unwrap(), 2);
            let first_start = records_offset + record.pos;
            assert!(decode_v2_fusion_tree_pair_key(&mut record)
                .unwrap()
                .is_some());
            let first_end = records_offset + record.pos;
            let second_start = first_end;
            assert!(decode_v2_fusion_tree_pair_key(&mut record)
                .unwrap()
                .is_some());
            let second_end = records_offset + record.pos;
            (first_start..first_end, second_start..second_end)
        };
        assert_eq!(first_src.len(), second_src.len());
        let first_src_bytes = duplicate_bytes[first_src].to_vec();
        duplicate_bytes[second_src].copy_from_slice(&first_src_bytes);
        let (categorical_key, categorical_plan, categorical_bytes) = categorical_v2_fixture();
        let mut mixed = categorical_bytes[..TREE_PLAN_CACHE_MAGIC.len() + 8].to_vec();
        write_usize(&mut mixed, 2);
        mixed.extend_from_slice(&duplicate_bytes[records_offset..]);
        mixed.extend_from_slice(&categorical_bytes[records_offset..]);

        let decoded = decode_builtin_tree_plan_cache(&mixed).unwrap();

        // What: duplicate source identities make a persisted cache key
        // non-canonical, but its paired plan is still consumed before replay
        // resumes at the next record.
        assert_eq!(decoded, vec![(categorical_key, categorical_plan)]);
    }

    #[test]
    fn source_group_key_rejects_repeated_public_group_indices() {
        let key = multiplicity_free_key();
        let structure =
            crate::tests::packed_fixture_structure(2, [(key.clone(), vec![1, 1])]).unwrap();
        let repeated = FusionTreeBlockGroup::new(key.group_key(), vec![0, 0]);

        let err = TreeTransformSourceGroupKey::from_group(&structure, &repeated).unwrap_err();

        // What: a caller-built group cannot create a non-canonical cache key by
        // repeating one valid source basis index.
        assert_eq!(
            err,
            OperationError::DuplicateTreeTransformKey {
                tensor: "src",
                index: 1,
            }
        );
    }

    #[test]
    fn persistent_builtin_tree_plan_cache_round_trips_with_version_guard() {
        let group_key = FusionTreeGroupKey::from_sector_ids([1, 1], [], [false, false], []);
        let key = TreeTransformSectorPlanKey {
            rule: TreeTransformBuiltinRuleCacheKey::SU2Exact {
                authority_version: tenet_core::SU2_EXACT_AUTHORITY_VERSION,
            },
            scope: TreeTransformPlanScope::TreePair,
            operation: TreeTransformOperation::braid([0, 1], [], [3, 5], []),
            source_groups: vec![TreeTransformSourceGroupKey {
                group_key: group_key.clone(),
                src_keys: vec![multiplicity_free_key()],
            }],
        };
        fn assert_categorical_keys(_: &[FusionTreePairKey]) {}
        assert_categorical_keys(key.source_groups()[0].src_keys());
        let categorical_key = multiplicity_free_key();
        let legacy_single = TreeTransformGroupBlockSpec::try_multi(
            [categorical_key.clone()],
            [categorical_key.clone()],
            vec![1.25],
        )
        .unwrap()
        .with_source_axes([0, 1]);
        let legacy_second = TreeTransformGroupBlockSpec::try_multi(
            [categorical_key.clone()],
            [categorical_key.clone()],
            vec![-2.5],
        )
        .unwrap()
        .with_source_axes([0, 1]);
        let plan = Arc::new(TreeTransformGroupPlan::new(vec![
            legacy_single,
            legacy_second,
        ]));
        let current_plan = Arc::new(TreeTransformGroupPlan::new(vec![
            TreeTransformGroupBlockSpec::single(
                categorical_key.clone(),
                categorical_key.clone(),
                1.25,
            )
            .with_source_axes([0, 1]),
            TreeTransformGroupBlockSpec::single(categorical_key.clone(), categorical_key, -2.5)
                .with_source_axes([0, 1]),
        ]));
        let mut plans = FxHashMap::default();
        plans.insert(key.clone(), Arc::clone(&plan));

        let mut bytes = encode_builtin_tree_plan_cache(&plans).unwrap();
        assert_eq!(
            &bytes[TREE_PLAN_CACHE_MAGIC.len()..TREE_PLAN_CACHE_MAGIC.len() + 8],
            &TREE_PLAN_CACHE_VERSION.to_le_bytes()
        );
        let decoded = decode_builtin_tree_plan_cache(&bytes).unwrap();
        assert_eq!(decoded, vec![(key, (*plan).clone())]);
        // What: categorical v2 bytes preserve canonical Single lowering and
        // share equal source-axis storage after decoding.
        assert!(decoded[0]
            .1
            .specs()
            .iter()
            .all(|spec| format!("{spec:?}").contains("Single")));
        assert!(std::ptr::eq(
            decoded[0].1.specs()[0].source_axes().unwrap(),
            decoded[0].1.specs()[1].source_axes().unwrap(),
        ));
        let mut current_plans = FxHashMap::default();
        current_plans.insert(decoded[0].0.clone(), current_plan);
        assert_eq!(
            encode_builtin_tree_plan_cache(&current_plans).unwrap(),
            bytes
        );
        let mut decoded_plans = FxHashMap::default();
        decoded_plans.insert(decoded[0].0.clone(), Arc::new(decoded[0].1.clone()));
        assert_eq!(
            encode_builtin_tree_plan_cache(&decoded_plans).unwrap(),
            bytes
        );

        let version_offset = TREE_PLAN_CACHE_MAGIC.len();
        bytes[version_offset..version_offset + 8].copy_from_slice(&1_u64.to_le_bytes());
        assert!(decode_builtin_tree_plan_cache(&bytes).is_err());

        let mut bytes = encode_builtin_tree_plan_cache(&plans).unwrap();
        let authority_offset = TREE_PLAN_CACHE_MAGIC.len() + 8 + 8 + 1;
        bytes[authority_offset] = tenet_core::SU2_EXACT_AUTHORITY_VERSION + 1;
        assert!(decode_builtin_tree_plan_cache(&bytes).is_err());
    }

    #[test]
    fn persistent_v2_omits_generic_tree_identity_without_mutating_memory() {
        let rule = GenericCacheProbeRule;
        let generic_codomain = FusionTreeKey::try_new_for_rule(
            &rule,
            [SectorId::new(0)],
            Some(SectorId::new(0)),
            [false],
            [],
            [],
        )
        .unwrap();
        let generic_domain = FusionTreeKey::try_new_for_rule(&rule, [], None, [], [], []).unwrap();
        let generic_key = FusionTreePairKey::pair(generic_codomain, generic_domain);
        let group_key = FusionTreeGroupKey::from_sector_ids([0], [], [false], []);
        let operation = TreeTransformOperation::braid([0], [], [0], []);
        let generic_plan_key = TreeTransformSectorPlanKey {
            rule: TreeTransformBuiltinRuleCacheKey::SU2Exact {
                authority_version: tenet_core::SU2_EXACT_AUTHORITY_VERSION,
            },
            scope: TreeTransformPlanScope::TreePair,
            operation: operation.clone(),
            source_groups: vec![TreeTransformSourceGroupKey {
                group_key: group_key.clone(),
                src_keys: vec![generic_key.clone()],
            }],
        };
        let generic_plan = Arc::new(TreeTransformGroupPlan::new(vec![
            TreeTransformGroupBlockSpec::single(generic_key.clone(), generic_key, 1.0),
        ]));
        let multiplicity_free_key = multiplicity_free_key();
        let multiplicity_free_plan_key = TreeTransformSectorPlanKey {
            rule: TreeTransformBuiltinRuleCacheKey::SU2Exact {
                authority_version: tenet_core::SU2_EXACT_AUTHORITY_VERSION,
            },
            scope: TreeTransformPlanScope::TreePair,
            operation,
            source_groups: vec![TreeTransformSourceGroupKey {
                group_key: multiplicity_free_key.group_key(),
                src_keys: vec![multiplicity_free_key.clone()],
            }],
        };
        let multiplicity_free_plan = Arc::new(TreeTransformGroupPlan::new(vec![
            TreeTransformGroupBlockSpec::single(
                multiplicity_free_key.clone(),
                multiplicity_free_key,
                1.0,
            ),
        ]));
        let mut plans = FxHashMap::default();
        plans.insert(generic_plan_key.clone(), generic_plan);
        plans.insert(multiplicity_free_plan_key.clone(), multiplicity_free_plan);

        let encoded = encode_builtin_tree_plan_cache(&plans).unwrap();
        let decoded = decode_builtin_tree_plan_cache(&encoded)
            .expect("multiplicity-free persistent entry decodes");

        // What: v2 cannot encode the outer-multiplicity identity bit, so only
        // the multiplicity-free entry crosses the persistent boundary.
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].0, multiplicity_free_plan_key);
        assert_eq!(plans.len(), 2);
        assert!(plans.contains_key(&generic_plan_key));
    }

    #[test]
    fn legacy_v2_pre_split_sector_key_bytes_remain_categorical_without_opaque_aliasing() {
        // What: literal v2 bytes emitted for the old
        // `BlockKey::sector_ids([1, 1])` representation. Before namespace
        // separation that application label was serialized as a malformed
        // categorical pair: two codomain sectors, no coupled sector, no
        // vertices, and an empty domain tree.
        const PRE_SPLIT_FAKE_CATEGORICAL_KEY: &str = concat!(
            "01",
            "0200000000000000",
            "0100000000000000",
            "0100000000000000",
            "00",
            "0200000000000000",
            "0000",
            "0000000000000000",
            "0000000000000000",
            "0000000000000000",
            "00",
            "0000000000000000",
            "0000000000000000",
            "0000000000000000",
        );
        let bytes = decode_hex_fixture(PRE_SPLIT_FAKE_CATEGORICAL_KEY);
        let mut input = CacheBytes::new(&bytes);
        let decoded = decode_v2_fusion_tree_pair_key(&mut input)
            .unwrap()
            .expect("legacy tag 1 remains categorical");
        input.finish().unwrap();

        assert_eq!(
            decoded.codomain_uncoupled(),
            &[SectorId::new(1), SectorId::new(1)]
        );
        assert_eq!(decoded.coupled(), None);
        assert!(decoded.codomain_vertices().is_empty());
        assert!(decoded
            .validate_for_rule(&tenet_core::Z2FusionRule)
            .is_err());
        assert_ne!(BlockKey::from(decoded), BlockKey::opaque([1, 1]));
    }
}

pub type TreePairTransformCache<T, RuleKey> = TreeTransformCache<T, RuleKey>;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TreeTransformCacheStats {
    plan_hits: usize,
    plan_misses: usize,
    structure_hits: usize,
    structure_misses: usize,
    tree_row_hits: usize,
    tree_row_misses: usize,
}

impl TreeTransformCacheStats {
    #[inline]
    pub fn plan_hits(self) -> usize {
        self.plan_hits
    }

    /// Shape-independent recoupling-row memo hits (TensorKit
    /// fstranspose/fsbraid @cached analog): rows reused across structure changes.
    #[inline]
    pub fn tree_row_hits(self) -> usize {
        self.tree_row_hits
    }

    #[inline]
    pub fn tree_row_misses(self) -> usize {
        self.tree_row_misses
    }

    #[inline]
    pub fn plan_misses(self) -> usize {
        self.plan_misses
    }

    #[inline]
    pub fn structure_hits(self) -> usize {
        self.structure_hits
    }

    #[inline]
    pub fn structure_misses(self) -> usize {
        self.structure_misses
    }
}

#[derive(Clone, Debug)]
struct TreeTransformLastStructure<T, RuleKey> {
    plan_key: TreeTransformSectorPlanKey<RuleKey>,
    structure_key: TreeTransformStructureCacheKey<TreeTransformSectorPlanKey<RuleKey>>,
    rule: RuleKey,
    scope: TreeTransformPlanScope,
    operation: TreeTransformOperation,
    // `dst_ptr`/`src_ptr` are raw-pointer keys, sound only because the payload
    // `structure: Arc<TreeTransformStructure<T>>` transitively pins the dst/src
    // structures it was built from (it owns `dst_structure`/`src_structure`
    // Arcs — see transform_structure.rs). While this entry lives, those
    // addresses cannot be recycled, so a pointer match is a true identity match.
    // This safety depends on that payload pinning: if `TreeTransformStructure`
    // ever stopped holding those Arcs, these keys would become unsound (ABA).
    dst_ptr: usize,
    src_ptr: usize,
    storage_conjugate: bool,
    structure: Arc<TreeTransformStructure<T>>,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct TreeTransformFastStructureKey<RuleKey> {
    rule: RuleKey,
    scope: TreeTransformPlanScope,
    operation: TreeTransformOperation,
    dst_structure_id: usize,
    src_structure_id: usize,
    storage_conjugate: bool,
}

impl<T, RuleKey> Default for TreeTransformCache<T, RuleKey> {
    fn default() -> Self {
        Self {
            plans: FxHashMap::default(),
            plan_lru_order: VecDeque::new(),
            structures: TreeTransformStructureCache::default(),
            last_structure: None,
            fast_structures: FxHashMap::default(),
            policy: OperationCachePolicy::default(),
            stats: TreeTransformCacheStats::default(),
            tree_rows: crate::tree_transform::plan::TreePairRowMemo::default(),
            all_codomain_rows: crate::tree_transform::plan::AllCodomainRowMemo::default(),
            recoupling_threads: 1,
        }
    }
}

impl<T, RuleKey> TreeTransformCache<T, RuleKey>
where
    RuleKey: Clone + Eq + Hash,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_policy(policy: OperationCachePolicy) -> Self {
        Self {
            plans: FxHashMap::default(),
            plan_lru_order: VecDeque::new(),
            structures: TreeTransformStructureCache::with_policy(policy),
            last_structure: None,
            fast_structures: FxHashMap::default(),
            policy,
            stats: TreeTransformCacheStats::default(),
            tree_rows: crate::tree_transform::plan::TreePairRowMemo::default(),
            all_codomain_rows: crate::tree_transform::plan::AllCodomainRowMemo::default(),
            recoupling_threads: 1,
        }
    }

    #[inline]
    pub fn recoupling_threads(&self) -> usize {
        self.recoupling_threads
    }

    /// Plan-compile worker count; the execution context keeps this in sync
    /// with the backend's `recoupling_threads`, so the one configured knob
    /// drives both replay and compile parallelism. Serial and parallel
    /// compilation use the same staged group algorithm.
    pub fn set_recoupling_threads(&mut self, threads: usize) {
        self.recoupling_threads = threads.max(1);
    }

    #[inline]
    pub fn policy(&self) -> OperationCachePolicy {
        self.policy
    }

    pub fn set_policy(&mut self, policy: OperationCachePolicy) {
        self.policy = policy;
        self.structures.set_policy(policy);
        self.last_structure = None;
        self.fast_structures.clear();
        if !policy.stores_entries() {
            self.plans.clear();
            self.plan_lru_order.clear();
        } else if let Some(max_entries) = policy.max_entries() {
            rebuild_lru_order_from_keys(&self.plans, &mut self.plan_lru_order);
            self.enforce_plan_lru_limit(max_entries);
        }
    }

    #[inline]
    pub fn plan_len(&self) -> usize {
        self.plans.len()
    }

    #[inline]
    pub fn structure_len(&self) -> usize {
        self.structures.len()
    }

    #[cfg(test)]
    pub(crate) fn tree_row_len(&self) -> usize {
        self.tree_rows.len()
    }

    #[cfg(test)]
    pub(crate) fn all_codomain_row_len(&self) -> usize {
        self.all_codomain_rows.len()
    }

    #[cfg(test)]
    pub(crate) fn fast_structure_len(&self) -> usize {
        self.fast_structures.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.plans.is_empty() && self.structures.is_empty()
    }

    #[inline]
    pub fn stats(&self) -> TreeTransformCacheStats {
        self.stats
    }

    pub fn reset_stats(&mut self) {
        self.stats = TreeTransformCacheStats::default();
    }

    fn fast_structure(
        &mut self,
        rule_key: &RuleKey,
        scope: TreeTransformPlanScope,
        operation: &TreeTransformOperation,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        storage_conjugate: bool,
    ) -> Option<Arc<TreeTransformStructure<T>>> {
        if !self.policy.stores_entries() {
            return None;
        }
        let last = self.last_structure.as_ref()?;
        if &last.rule == rule_key
            && last.scope == scope
            && &last.operation == operation
            && last.dst_ptr == Arc::as_ptr(dst_structure) as usize
            && last.src_ptr == Arc::as_ptr(src_structure) as usize
            && last.storage_conjugate == storage_conjugate
        {
            let structure = Arc::clone(&last.structure);
            self.stats.plan_hits += 1;
            self.stats.structure_hits += 1;
            if self.policy.max_entries().is_some() {
                let plan_key = last.plan_key.clone();
                let structure_key = last.structure_key.clone();
                self.touch_plan(&plan_key);
                self.structures.touch(&structure_key);
            }
            Some(structure)
        } else {
            if self.policy.max_entries().is_some() {
                return None;
            }
            let key = TreeTransformFastStructureKey {
                rule: rule_key.clone(),
                scope,
                operation: operation.clone(),
                dst_structure_id: dst_structure.content_id(),
                src_structure_id: src_structure.content_id(),
                storage_conjugate,
            };
            let structure = self.fast_structures.get(&key)?;
            self.stats.plan_hits += 1;
            self.stats.structure_hits += 1;
            Some(Arc::clone(structure))
        }
    }

    fn remember_structure(
        &mut self,
        plan_key: TreeTransformSectorPlanKey<RuleKey>,
        structure_key: TreeTransformStructureCacheKey<TreeTransformSectorPlanKey<RuleKey>>,
        rule: RuleKey,
        scope: TreeTransformPlanScope,
        operation: TreeTransformOperation,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        storage_conjugate: bool,
        structure: Arc<TreeTransformStructure<T>>,
    ) {
        if self.policy.stores_entries() && self.policy.max_entries().is_none() {
            self.fast_structures.insert(
                TreeTransformFastStructureKey {
                    rule: rule.clone(),
                    scope,
                    operation: operation.clone(),
                    dst_structure_id: dst_structure.content_id(),
                    src_structure_id: src_structure.content_id(),
                    storage_conjugate,
                },
                Arc::clone(&structure),
            );
        }
        self.last_structure = Some(TreeTransformLastStructure {
            plan_key,
            structure_key,
            rule,
            scope,
            operation,
            dst_ptr: Arc::as_ptr(dst_structure) as usize,
            src_ptr: Arc::as_ptr(src_structure) as usize,
            storage_conjugate,
            structure,
        });
    }

    fn touch_plan(&mut self, key: &TreeTransformSectorPlanKey<RuleKey>) {
        if self.policy.max_entries().is_some() && self.plans.contains_key(key) {
            touch_lru_key(&mut self.plan_lru_order, key);
        }
    }

    fn insert_plan_arc(
        &mut self,
        key: TreeTransformSectorPlanKey<RuleKey>,
        plan: Arc<TreeTransformGroupPlan<T>>,
    ) {
        if !self.policy.stores_entries() {
            return;
        }
        self.plans.insert(key.clone(), plan);
        if self.policy.max_entries().is_some() {
            self.touch_plan(&key);
        }
        if let Some(max_entries) = self.policy.max_entries() {
            self.enforce_plan_lru_limit(max_entries);
        }
    }

    fn enforce_plan_lru_limit(&mut self, max_entries: usize) {
        while self.plans.len() > max_entries {
            let Some(oldest) = self.plan_lru_order.pop_front() else {
                break;
            };
            self.plans.remove(&oldest);
        }
    }

    fn get_or_compile_global_tree_pair_plan<R>(
        &mut self,
        source_proof: &LocallyValidatedFusionTreeBlockStructure<'_, '_, R>,
        rule_key: &RuleKey,
        operation: TreeTransformOperation,
        plan_key: &TreeTransformSectorPlanKey<RuleKey>,
    ) -> Result<Arc<TreeTransformGroupPlan<T>>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = T>,
        T: 'static + Clone + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync,
        RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
    {
        load_persistent_builtin_tree_plans_if_needed();
        let global_plans = global_tree_transform_plans::<T, RuleKey>();
        if let Some(plan) = global_plans
            .read()
            .expect("global tree-transform plan cache poisoned")
            .get(plan_key)
            .cloned()
        {
            return Ok(plan);
        }

        let global_rows = global_tree_pair_rows::<T, RuleKey>();
        {
            let rows = global_rows
                .read()
                .expect("global tree-pair row cache poisoned");
            for (key, value) in rows.iter() {
                self.tree_rows
                    .entry(key.clone())
                    .or_insert_with(|| Arc::clone(value));
            }
        }
        let plan =
            crate::tree_transform::plan::build_multiplicity_free_tree_pair_transform_group_plan_memoized_validated(
            source_proof,
            rule_key,
            operation,
            &mut self.tree_rows,
            &mut self.stats.tree_row_hits,
            &mut self.stats.tree_row_misses,
            self.recoupling_threads,
        )?;
        {
            let mut rows = global_rows
                .write()
                .expect("global tree-pair row cache poisoned");
            for (key, value) in self.tree_rows.iter() {
                rows.entry(key.clone()).or_insert_with(|| Arc::clone(value));
            }
        }

        let plan = Arc::new(plan);
        global_plans
            .write()
            .expect("global tree-transform plan cache poisoned")
            .entry(plan_key.clone())
            .or_insert_with(|| Arc::clone(&plan));
        persist_builtin_tree_plans_if_enabled();
        Ok(plan)
    }

    fn get_or_compile_global_all_codomain_plan<R>(
        &mut self,
        source_proof: &LocallyValidatedAllCodomainFusionTreeBlockStructure<'_, '_, R>,
        rule_key: &RuleKey,
        operation: TreeTransformOperation,
        plan_key: &TreeTransformSectorPlanKey<RuleKey>,
    ) -> Result<Arc<TreeTransformGroupPlan<T>>, OperationError>
    where
        R: MultiplicityFreeFusionSymbols<Scalar = T> + Sync,
        T: 'static + Clone + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync,
        RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
    {
        load_persistent_builtin_tree_plans_if_needed();
        let global_plans = global_tree_transform_plans::<T, RuleKey>();
        if let Some(plan) = global_plans
            .read()
            .expect("global tree-transform plan cache poisoned")
            .get(plan_key)
            .cloned()
        {
            return Ok(plan);
        }

        let global_rows = global_all_codomain_rows::<T, RuleKey>();
        {
            let rows = global_rows
                .read()
                .expect("global all-codomain row cache poisoned");
            for (key, value) in rows.iter() {
                self.all_codomain_rows
                    .entry(key.clone())
                    .or_insert_with(|| Arc::clone(value));
            }
        }
        let plan =
            crate::tree_transform::plan::build_multiplicity_free_all_codomain_tree_transform_group_plan_memoized_validated(
            source_proof,
            rule_key,
            operation,
            &mut self.all_codomain_rows,
            &mut self.stats.tree_row_hits,
            &mut self.stats.tree_row_misses,
            self.recoupling_threads,
        )?;
        {
            let mut rows = global_rows
                .write()
                .expect("global all-codomain row cache poisoned");
            for (key, value) in self.all_codomain_rows.iter() {
                rows.entry(key.clone()).or_insert_with(|| Arc::clone(value));
            }
        }

        let plan = Arc::new(plan);
        global_plans
            .write()
            .expect("global tree-transform plan cache poisoned")
            .entry(plan_key.clone())
            .or_insert_with(|| Arc::clone(&plan));
        persist_builtin_tree_plans_if_enabled();
        Ok(plan)
    }

    /// Resolve an exact tree-pair replay structure.
    ///
    /// Raw block keys follow
    /// [`FusionTreeKey::validate_for_rule`]'s provider-domain precondition.
    pub fn get_or_compile_tree_pair<
        R,
        TDst,
        TSrc,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const SRC_NOUT: usize,
        const SRC_NIN: usize,
        SDst,
        SSrc,
        DDst,
        DSrc,
    >(
        &mut self,
        rule: &R,
        operation: TreeTransformOperation,
        dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst, DDst>,
        src: &TensorMap<TSrc, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    ) -> Result<Arc<TreeTransformStructure<T>>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = T> + TreeTransformRuleCacheKey<Key = RuleKey>,
        T: 'static + Copy + Clone + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync,
        RuleKey: 'static + Send + Sync,
        DDst: TensorStorage<TDst>,
        DSrc: TensorStorage<TSrc>,
    {
        validate_multiplicity_free_tree_transform_capability(rule, &operation)?;
        validate_tree_pair_namespace_before_cache(&operation, src.structure())?;
        if rule.fusion_style() == tenet_core::FusionStyleKind::Unique {
            let source_proof = validate_multiplicity_free_tree_pair_preflight_after_capability(
                rule,
                &operation,
                src.structure(),
            )?;
            LocallyValidatedFusionTreeBlockStructure::try_new(rule, dst.structure())
                .map_err(OperationError::from_core_preserving_context)?;
            // Why-not cache Unique plans: Unique removes fusion multiplicity,
            // but does not imply a single total destination. Retaining
            // plan/row state costs more than direct compilation here and
            // risks process-lifetime growth for cheap keys.
            self.stats.plan_misses += 1;
            self.stats.structure_misses += 1;
            let plan =
                build_tree_pair_transform_group_plan_validated(&source_proof, operation.clone())?;
            return Ok(Arc::new(plan.compile(dst, src)?));
        }
        let rule_key = rule.tree_transform_rule_cache_key();
        if let Some(structure) = self.fast_structure(
            &rule_key,
            TreeTransformPlanScope::TreePair,
            &operation,
            dst.structure(),
            src.structure(),
            false,
        ) {
            return Ok(structure);
        }
        let source_proof = validate_multiplicity_free_tree_pair_preflight_after_capability(
            rule,
            &operation,
            src.structure(),
        )?;
        LocallyValidatedFusionTreeBlockStructure::try_new(rule, dst.structure())
            .map_err(OperationError::from_core_preserving_context)?;
        let plan_key = TreeTransformSectorPlanKey::new(
            rule_key.clone(),
            TreeTransformPlanScope::TreePair,
            operation.clone(),
            src.structure(),
        )?;
        if !self.policy.stores_entries() {
            self.stats.plan_misses += 1;
            self.stats.structure_misses += 1;
            let plan =
                build_tree_pair_transform_group_plan_validated(&source_proof, operation.clone())?;
            return Ok(Arc::new(plan.compile(dst, src)?));
        }
        if self.plans.contains_key(&plan_key) {
            self.stats.plan_hits += 1;
            self.touch_plan(&plan_key);
        } else {
            self.stats.plan_misses += 1;
            let plan = self.get_or_compile_global_tree_pair_plan(
                &source_proof,
                &rule_key,
                operation.clone(),
                &plan_key,
            )?;
            self.insert_plan_arc(plan_key.clone(), plan);
        }
        self.get_or_compile_structure(
            rule_key,
            TreeTransformPlanScope::TreePair,
            operation,
            plan_key,
            dst,
            src,
        )
    }

    /// Structure-only variant of [`Self::get_or_compile_tree_pair`], with the
    /// same provider-domain precondition.
    pub fn get_or_compile_tree_pair_structures_with_storage_conjugation<R>(
        &mut self,
        rule: &R,
        operation: TreeTransformOperation,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        storage_conjugate: bool,
    ) -> Result<Arc<TreeTransformStructure<T>>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = T> + TreeTransformRuleCacheKey<Key = RuleKey>,
        T: 'static + Copy + Clone + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync,
        RuleKey: 'static + Send + Sync,
    {
        self.get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
            rule,
            &operation,
            dst_structure,
            src_structure,
            storage_conjugate,
        )
    }

    /// Borrowed-operation variant of
    /// [`Self::get_or_compile_tree_pair_structures_with_storage_conjugation`],
    /// with the same provider-domain precondition.
    pub fn get_or_compile_tree_pair_structures_with_storage_conjugation_ref<R>(
        &mut self,
        rule: &R,
        operation: &TreeTransformOperation,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        storage_conjugate: bool,
    ) -> Result<Arc<TreeTransformStructure<T>>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = T> + TreeTransformRuleCacheKey<Key = RuleKey>,
        T: 'static + Copy + Clone + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync,
        RuleKey: 'static + Send + Sync,
    {
        validate_multiplicity_free_tree_transform_capability(rule, operation)?;
        validate_tree_pair_namespace_before_cache(operation, src_structure)?;
        if rule.fusion_style() == tenet_core::FusionStyleKind::Unique {
            let source_proof = validate_multiplicity_free_tree_pair_preflight_after_capability(
                rule,
                operation,
                src_structure,
            )?;
            LocallyValidatedFusionTreeBlockStructure::try_new(rule, dst_structure)
                .map_err(OperationError::from_core_preserving_context)?;
            // Why-not cache Unique plans: Unique removes fusion multiplicity,
            // but does not imply a single total destination. The storage-only
            // form still has cheap, layout-specific entries, so retaining a
            // second structural cache would duplicate state and permit
            // unbounded entries.
            self.stats.plan_misses += 1;
            self.stats.structure_misses += 1;
            let plan =
                build_tree_pair_transform_group_plan_validated(&source_proof, operation.clone())?;
            return Ok(Arc::new(
                plan.compile_shared_structures_with_storage_conjugation(
                    Arc::clone(dst_structure),
                    Arc::clone(src_structure),
                    storage_conjugate,
                )?,
            ));
        }
        let rule_key = rule.tree_transform_rule_cache_key();
        if let Some(structure) = self.fast_structure(
            &rule_key,
            TreeTransformPlanScope::TreePair,
            operation,
            dst_structure,
            src_structure,
            storage_conjugate,
        ) {
            return Ok(structure);
        }
        let source_proof = validate_multiplicity_free_tree_pair_preflight_after_capability(
            rule,
            operation,
            src_structure,
        )?;
        LocallyValidatedFusionTreeBlockStructure::try_new(rule, dst_structure)
            .map_err(OperationError::from_core_preserving_context)?;
        let plan_key = TreeTransformSectorPlanKey::new(
            rule_key.clone(),
            TreeTransformPlanScope::TreePair,
            operation.clone(),
            src_structure,
        )?;
        if !self.policy.stores_entries() {
            self.stats.plan_misses += 1;
            self.stats.structure_misses += 1;
            let plan =
                build_tree_pair_transform_group_plan_validated(&source_proof, operation.clone())?;
            return Ok(Arc::new(
                plan.compile_shared_structures_with_storage_conjugation(
                    Arc::clone(dst_structure),
                    Arc::clone(src_structure),
                    storage_conjugate,
                )?,
            ));
        }
        if self.plans.contains_key(&plan_key) {
            self.stats.plan_hits += 1;
            self.touch_plan(&plan_key);
        } else {
            self.stats.plan_misses += 1;
            let plan = self.get_or_compile_global_tree_pair_plan(
                &source_proof,
                &rule_key,
                operation.clone(),
                &plan_key,
            )?;
            self.insert_plan_arc(plan_key.clone(), plan);
        }
        self.get_or_compile_structure_from_structures_with_storage_conjugation(
            rule_key,
            TreeTransformPlanScope::TreePair,
            operation.clone(),
            plan_key,
            dst_structure,
            src_structure,
            storage_conjugate,
        )
    }

    pub(crate) fn get_or_compile_tree_pair_prelowered<R, FBlock, FAxis>(
        &mut self,
        rule: &R,
        operation: &TreeTransformOperation,
        dst_structure: &Arc<BlockStructure>,
        logical_src_structure: &Arc<BlockStructure>,
        storage_src_structure: &Arc<BlockStructure>,
        storage_conjugate: bool,
        logical_to_storage_block: FBlock,
        logical_to_storage_axis: FAxis,
    ) -> Result<Arc<TreeTransformStructure<T>>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = T> + TreeTransformRuleCacheKey<Key = RuleKey>,
        T: 'static + Copy + Clone + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync,
        RuleKey: 'static + Send + Sync,
        FBlock: Fn(usize) -> Result<usize, OperationError>,
        FAxis: Fn(usize) -> Result<usize, OperationError>,
    {
        let source_proof =
            validate_multiplicity_free_tree_pair_preflight(rule, operation, logical_src_structure)?;
        let logical_source_id = logical_src_structure.content_id();
        let storage_source_id = storage_src_structure.content_id();
        let destination_id = dst_structure.content_id();
        if storage_source_id != logical_source_id {
            LocallyValidatedFusionTreeBlockStructure::try_new(rule, storage_src_structure)
                .map_err(OperationError::from_core_preserving_context)?;
        }
        if destination_id != logical_source_id && destination_id != storage_source_id {
            LocallyValidatedFusionTreeBlockStructure::try_new(rule, dst_structure)
                .map_err(OperationError::from_core_preserving_context)?;
        }
        let rule_key = rule.tree_transform_rule_cache_key();
        let plan_key = TreeTransformSectorPlanKey::new(
            rule_key.clone(),
            TreeTransformPlanScope::TreePair,
            operation.clone(),
            logical_src_structure,
        )?;
        if !self.policy.stores_entries() {
            self.stats.plan_misses += 1;
            self.stats.structure_misses += 1;
            let plan =
                build_tree_pair_transform_group_plan_validated(&source_proof, operation.clone())?;
            return Ok(Arc::new(
                plan.compile_shared_structures_with_storage_mapping(
                    Arc::clone(dst_structure),
                    logical_src_structure,
                    Arc::clone(storage_src_structure),
                    logical_to_storage_block,
                    logical_to_storage_axis,
                    storage_conjugate,
                )?,
            ));
        }
        if self.plans.contains_key(&plan_key) {
            self.stats.plan_hits += 1;
            self.touch_plan(&plan_key);
        } else {
            self.stats.plan_misses += 1;
            let plan = self.get_or_compile_global_tree_pair_plan(
                &source_proof,
                &rule_key,
                operation.clone(),
                &plan_key,
            )?;
            self.insert_plan_arc(plan_key.clone(), plan);
        }
        let structure_key =
            TreeTransformStructureCacheKey::from_structures_with_storage_conjugation(
                plan_key.clone(),
                dst_structure,
                storage_src_structure,
                storage_conjugate,
            )?;
        if let Some(structure) = self.structures.get_arc(&structure_key) {
            self.stats.structure_hits += 1;
            self.structures.touch(&structure_key);
            return Ok(structure);
        }
        self.stats.structure_misses += 1;
        let plan = self
            .plans
            .get(&plan_key)
            .expect("tree transform plan inserted before prelowered structure compile");
        let structure = Arc::new(plan.compile_shared_structures_with_storage_mapping(
            Arc::clone(dst_structure),
            logical_src_structure,
            Arc::clone(storage_src_structure),
            logical_to_storage_block,
            logical_to_storage_axis,
            storage_conjugate,
        )?);
        self.structures
            .insert_arc(structure_key, Arc::clone(&structure));
        Ok(structure)
    }

    /// Generic-fusion (outer-multiplicity, e.g. SU(3)) sibling of
    /// [`Self::get_or_compile_tree_pair`].
    ///
    /// ponytail: NON-MEMOIZED to start — it rebuilds and compiles the generic
    /// plan on every call. Generic (SU(3)) recoupling is not yet on any hot
    /// path, and correctness-before-perf is the Stage B rule; the memoized
    /// builder (the generic analogue of `TreePairRowMemo` / the plan/row cache
    /// the mult-free sibling above uses) is deferred until a real Generic
    /// workload measures the recompile cost (the B3c / perf handoff). The
    /// `TreeTransformRuleCacheKey` bound is still required: the Su3 `Key` embeds
    /// the table's provenance hash, so once memoization lands a swapped table
    /// can never reuse another table's plans.
    ///
    /// Raw block keys follow
    /// [`FusionTreeKey::validate_for_rule`]'s provider-domain precondition.
    pub fn get_or_compile_tree_pair_generic<
        R,
        TDst,
        TSrc,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const SRC_NOUT: usize,
        const SRC_NIN: usize,
        SDst,
        SSrc,
        DDst,
        DSrc,
    >(
        &mut self,
        rule: &R,
        operation: TreeTransformOperation,
        dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst, DDst>,
        src: &TensorMap<TSrc, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    ) -> Result<Arc<TreeTransformStructure<T>>, OperationError>
    where
        R: GenericRigidSymbols<Scalar = T> + TreeTransformRuleCacheKey<Key = RuleKey>,
        R::Scalar: GenericBraidScalar,
        T: 'static + Copy + Clone + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync,
        RuleKey: 'static + Send + Sync,
        DDst: TensorStorage<TDst>,
        DSrc: TensorStorage<TSrc>,
    {
        let source_proof = validate_generic_tree_pair_preflight(rule, &operation, src.structure())?;
        let _destination_proof =
            LocallyValidatedFusionTreeBlockStructure::try_new(rule, dst.structure())
                .map_err(OperationError::from_core_preserving_context)?;
        self.stats.plan_misses += 1;
        self.stats.structure_misses += 1;
        let plan =
            build_generic_tree_pair_transform_group_plan_validated(&source_proof, operation)?;
        Ok(Arc::new(plan.compile(dst, src)?))
    }

    /// Structure-only generic sibling for the dynamic-rank (raw-slice) path —
    /// the top-level `tenet::Tensor` SU(3) `permute`/`braid`/`transpose` route.
    /// Same non-memoized rationale as [`Self::get_or_compile_tree_pair_generic`].
    /// It has the same provider-domain precondition.
    pub fn get_or_compile_tree_pair_structures_generic<R>(
        &mut self,
        rule: &R,
        operation: TreeTransformOperation,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
    ) -> Result<Arc<TreeTransformStructure<T>>, OperationError>
    where
        R: GenericRigidSymbols<Scalar = T> + TreeTransformRuleCacheKey<Key = RuleKey>,
        R::Scalar: GenericBraidScalar,
        T: 'static + Copy + Clone + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync,
        RuleKey: 'static + Send + Sync,
    {
        let source_proof = validate_generic_tree_pair_preflight(rule, &operation, src_structure)?;
        let _destination_proof =
            LocallyValidatedFusionTreeBlockStructure::try_new(rule, dst_structure)
                .map_err(OperationError::from_core_preserving_context)?;
        self.stats.plan_misses += 1;
        self.stats.structure_misses += 1;
        let plan =
            build_generic_tree_pair_transform_group_plan_validated(&source_proof, operation)?;
        Ok(Arc::new(
            plan.compile_shared_structures_with_storage_conjugation(
                Arc::clone(dst_structure),
                Arc::clone(src_structure),
                false,
            )?,
        ))
    }

    /// Resolve an exact all-codomain replay structure.
    ///
    /// Raw block keys follow
    /// [`FusionTreeKey::validate_for_rule`]'s provider-domain precondition.
    pub fn get_or_compile_all_codomain<
        R,
        TDst,
        TSrc,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const SRC_NOUT: usize,
        const SRC_NIN: usize,
        SDst,
        SSrc,
        DDst,
        DSrc,
    >(
        &mut self,
        rule: &R,
        operation: TreeTransformOperation,
        dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst, DDst>,
        src: &TensorMap<TSrc, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    ) -> Result<Arc<TreeTransformStructure<T>>, OperationError>
    where
        R: MultiplicityFreeFusionSymbols<Scalar = T>
            + TreeTransformRuleCacheKey<Key = RuleKey>
            + Sync,
        T: 'static + Copy + Clone + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync,
        RuleKey: 'static + Send + Sync,
        DDst: TensorStorage<TDst>,
        DSrc: TensorStorage<TSrc>,
    {
        validate_multiplicity_free_tree_transform_capability(rule, &operation)?;
        validate_all_codomain_namespace_before_cache(&operation, src.structure())?;
        if rule.fusion_style() == tenet_core::FusionStyleKind::Unique {
            let source_proof = validate_multiplicity_free_all_codomain_preflight_after_capability(
                rule,
                &operation,
                src.structure(),
            )?;
            LocallyValidatedFusionTreeBlockStructure::try_new(rule, dst.structure())
                .map_err(OperationError::from_core_preserving_context)?;
            // Why-not cache Unique all-codomain transforms: with no fusion
            // multiplicity this lowering is a direct tree relabeling.  A
            // process/context cache only retains layout descriptors for a
            // cheap, non-reusable key and defeats the eager Unique path used
            // by tree-pair transforms.
            self.stats.plan_misses += 1;
            self.stats.structure_misses += 1;
            let plan = build_all_codomain_tree_transform_group_plan_validated(
                &source_proof,
                operation.clone(),
            )?;
            return Ok(Arc::new(plan.compile(dst, src)?));
        }
        let rule_key = rule.tree_transform_rule_cache_key();
        if let Some(structure) = self.fast_structure(
            &rule_key,
            TreeTransformPlanScope::AllCodomain,
            &operation,
            dst.structure(),
            src.structure(),
            false,
        ) {
            return Ok(structure);
        }
        let source_proof = validate_multiplicity_free_all_codomain_preflight_after_capability(
            rule,
            &operation,
            src.structure(),
        )?;
        LocallyValidatedFusionTreeBlockStructure::try_new(rule, dst.structure())
            .map_err(OperationError::from_core_preserving_context)?;
        let plan_key = TreeTransformSectorPlanKey::new(
            rule_key.clone(),
            TreeTransformPlanScope::AllCodomain,
            operation.clone(),
            src.structure(),
        )?;
        if !self.policy.stores_entries() {
            self.stats.plan_misses += 1;
            self.stats.structure_misses += 1;
            let plan = build_all_codomain_tree_transform_group_plan_validated(
                &source_proof,
                operation.clone(),
            )?;
            return Ok(Arc::new(plan.compile(dst, src)?));
        }
        if self.plans.contains_key(&plan_key) {
            self.stats.plan_hits += 1;
            self.touch_plan(&plan_key);
        } else {
            self.stats.plan_misses += 1;
            let plan = self.get_or_compile_global_all_codomain_plan(
                &source_proof,
                &rule_key,
                operation.clone(),
                &plan_key,
            )?;
            self.insert_plan_arc(plan_key.clone(), plan);
        }
        self.get_or_compile_structure(
            rule_key,
            TreeTransformPlanScope::AllCodomain,
            operation,
            plan_key,
            dst,
            src,
        )
    }

    fn get_or_compile_structure<
        TDst,
        TSrc,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const SRC_NOUT: usize,
        const SRC_NIN: usize,
        SDst,
        SSrc,
        DDst,
        DSrc,
    >(
        &mut self,
        rule_key: RuleKey,
        scope: TreeTransformPlanScope,
        operation: TreeTransformOperation,
        plan_key: TreeTransformSectorPlanKey<RuleKey>,
        dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst, DDst>,
        src: &TensorMap<TSrc, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    ) -> Result<Arc<TreeTransformStructure<T>>, OperationError>
    where
        T: 'static + Copy + Send + Sync,
        RuleKey: 'static + Send + Sync,
        DDst: TensorStorage<TDst>,
        DSrc: TensorStorage<TSrc>,
    {
        let structure_key = TreeTransformStructureCacheKey::from_structures(
            plan_key.clone(),
            dst.structure(),
            src.structure(),
        )?;
        if self.structures.get(&structure_key).is_some() {
            self.stats.structure_hits += 1;
            self.structures.touch(&structure_key);
        } else {
            self.stats.structure_misses += 1;
            let global_structures = global_tree_transform_structures::<T, RuleKey>();
            let structure = global_structures
                .read()
                .expect("global tree-transform structure cache poisoned")
                .get(&structure_key)
                .cloned();
            if let Some(structure) = structure {
                self.structures
                    .insert_arc(structure_key.clone(), Arc::clone(&structure));
            } else {
                let plan = self
                    .plans
                    .get(&plan_key)
                    .expect("tree transform plan inserted before structure compile");
                let structure = Arc::new(plan.compile(dst, src)?);
                global_structures
                    .write()
                    .expect("global tree-transform structure cache poisoned")
                    .entry(structure_key.clone())
                    .or_insert_with(|| Arc::clone(&structure));
                self.structures.insert_arc(structure_key.clone(), structure);
            }
        }
        let structure = self
            .structures
            .get_arc(&structure_key)
            .expect("tree transform structure inserted before return");
        self.remember_structure(
            plan_key,
            structure_key,
            rule_key,
            scope,
            operation,
            dst.structure(),
            src.structure(),
            false,
            Arc::clone(&structure),
        );
        Ok(structure)
    }

    fn get_or_compile_structure_from_structures_with_storage_conjugation(
        &mut self,
        rule_key: RuleKey,
        scope: TreeTransformPlanScope,
        operation: TreeTransformOperation,
        plan_key: TreeTransformSectorPlanKey<RuleKey>,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        storage_conjugate: bool,
    ) -> Result<Arc<TreeTransformStructure<T>>, OperationError>
    where
        T: 'static + Copy + Send + Sync,
        RuleKey: 'static + Send + Sync,
    {
        let structure_key = TreeTransformStructureCacheKey::from_structures(
            plan_key.clone(),
            dst_structure,
            src_structure,
        )?;
        let structure_key = if storage_conjugate {
            TreeTransformStructureCacheKey::from_structures_with_storage_conjugation(
                plan_key.clone(),
                dst_structure,
                src_structure,
                true,
            )?
        } else {
            structure_key
        };
        if self.structures.get(&structure_key).is_some() {
            self.stats.structure_hits += 1;
            self.structures.touch(&structure_key);
        } else {
            self.stats.structure_misses += 1;
            let global_structures = global_tree_transform_structures::<T, RuleKey>();
            let structure = global_structures
                .read()
                .expect("global tree-transform structure cache poisoned")
                .get(&structure_key)
                .cloned();
            if let Some(structure) = structure {
                self.structures
                    .insert_arc(structure_key.clone(), Arc::clone(&structure));
            } else {
                let plan = self
                    .plans
                    .get(&plan_key)
                    .expect("tree transform plan inserted before structure compile");
                let structure = Arc::new(plan.compile_shared_structures_with_storage_conjugation(
                    Arc::clone(dst_structure),
                    Arc::clone(src_structure),
                    storage_conjugate,
                )?);
                global_structures
                    .write()
                    .expect("global tree-transform structure cache poisoned")
                    .entry(structure_key.clone())
                    .or_insert_with(|| Arc::clone(&structure));
                self.structures.insert_arc(structure_key.clone(), structure);
            }
        }
        let structure = self
            .structures
            .get_arc(&structure_key)
            .expect("tree transform structure inserted before return");
        self.remember_structure(
            plan_key,
            structure_key,
            rule_key,
            scope,
            operation,
            dst_structure,
            src_structure,
            storage_conjugate,
            Arc::clone(&structure),
        );
        Ok(structure)
    }
}
