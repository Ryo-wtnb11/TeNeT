use core::ops::{Add, Mul};
use std::collections::VecDeque;

use rustc_hash::FxHashMap;
use std::fs;
use std::hash::Hash;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use num_traits::Zero;
use tenet_core::{
    BlockKey, BlockStructure, FusionTreeBlockGroup, FusionTreeBlockKey, FusionTreeGroupKey,
    FusionTreeKey, GenericBraidScalar, GenericRigidSymbols, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols, SectorId, TensorMap, TensorStorage,
};

use crate::cache::{
    rebuild_lru_order_from_keys, touch_lru_key, typed_global_map, OperationCachePolicy,
    TreeTransformStructureCacheKey,
};
use crate::{OperationError, TreeTransformStructure, TreeTransformStructureCache};

use super::helpers::fusion_tree_group_block_keys;
use super::operation::{
    TreeTransformBuiltinRuleCacheKey, TreeTransformOperation, TreeTransformRuleCacheKey,
};
use super::plan::{
    build_generic_tree_pair_transform_group_plan, build_tree_pair_transform_group_plan,
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
    src_keys: Vec<BlockKey>,
}

impl TreeTransformSourceGroupKey {
    fn from_group(
        structure: &BlockStructure,
        group: &FusionTreeBlockGroup,
    ) -> Result<Self, OperationError> {
        Ok(Self {
            group_key: group.group_key().clone(),
            src_keys: fusion_tree_group_block_keys(structure, group, "src")?,
        })
    }

    #[inline]
    pub fn group_key(&self) -> &FusionTreeGroupKey {
        &self.group_key
    }

    #[inline]
    pub fn src_keys(&self) -> &[BlockKey] {
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
    dst_keys: Vec<BlockKey>,
    src_keys: Vec<BlockKey>,
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
    // Shape-independent row front for this context. Miss compiles prefill it
    // from the process-global row memo, then publish new rows back.
    tree_rows: crate::tree_transform::plan::TreePairRowMemo<T, RuleKey>,
    // Same row-granular memo for all-codomain transforms, keyed only by the
    // codomain tree because the domain is unchanged by this scope.
    all_codomain_rows: crate::tree_transform::plan::AllCodomainRowMemo<T, RuleKey>,
    // Worker count for plan compilation (missing tree-row computation).
    // Not a second knob: the execution context propagates the backend's
    // `recoupling_threads` here, so one setting drives replay and compile.
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

const TREE_PLAN_CACHE_MAGIC: &[u8] = b"TENET_TREE_PLAN_CACHE";
const TREE_PLAN_CACHE_VERSION: u64 = 1;
const TREE_PLAN_CACHE_FILE: &str = "tree_transform_plans_v1.bin";

fn persistent_tree_plan_cache_path() -> Option<PathBuf> {
    let dir = std::env::var_os("TENET_OPERATION_CACHE_DIR")?;
    if dir.is_empty() {
        return None;
    }
    Some(PathBuf::from(dir).join(TREE_PLAN_CACHE_FILE))
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
    let bytes = encode_builtin_tree_plan_cache(&plans);
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
) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(TREE_PLAN_CACHE_MAGIC);
    write_u64(&mut out, TREE_PLAN_CACHE_VERSION);
    write_usize(&mut out, plans.len());
    for (key, plan) in plans {
        encode_builtin_tree_plan_key(&mut out, key);
        encode_tree_transform_group_plan_f64(&mut out, plan);
    }
    out
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
        entries.push((key, plan));
    }
    input.finish()?;
    Ok(entries)
}

fn encode_builtin_tree_plan_key(
    out: &mut Vec<u8>,
    key: &TreeTransformSectorPlanKey<TreeTransformBuiltinRuleCacheKey>,
) {
    encode_builtin_rule_key(out, *key.rule());
    encode_plan_scope(out, key.scope());
    encode_tree_transform_operation(out, key.operation());
    write_usize(out, key.source_groups().len());
    for group in key.source_groups() {
        encode_fusion_tree_group_key(out, group.group_key());
        write_usize(out, group.src_keys().len());
        for key in group.src_keys() {
            encode_block_key(out, key);
        }
    }
}

fn decode_builtin_tree_plan_key(
    input: &mut CacheBytes<'_>,
) -> Result<TreeTransformSectorPlanKey<TreeTransformBuiltinRuleCacheKey>, ()> {
    let rule = decode_builtin_rule_key(input)?;
    let scope = decode_plan_scope(input)?;
    let operation = decode_tree_transform_operation(input)?;
    let group_count = input.read_usize()?;
    let mut source_groups = Vec::with_capacity(group_count);
    for _ in 0..group_count {
        let group_key = decode_fusion_tree_group_key(input)?;
        let key_count = input.read_usize()?;
        let mut src_keys = Vec::with_capacity(key_count);
        for _ in 0..key_count {
            src_keys.push(decode_block_key(input)?);
        }
        source_groups.push(TreeTransformSourceGroupKey {
            group_key,
            src_keys,
        });
    }
    Ok(TreeTransformSectorPlanKey {
        rule,
        scope,
        operation,
        source_groups,
    })
}

fn encode_tree_transform_group_plan_f64(out: &mut Vec<u8>, plan: &TreeTransformGroupPlan<f64>) {
    write_usize(out, plan.specs().len());
    for spec in plan.specs() {
        encode_fusion_tree_group_key(out, spec.group_key());
        write_usize(out, spec.dst_keys().len());
        for key in spec.dst_keys() {
            encode_block_key(out, key);
        }
        write_usize(out, spec.src_keys().len());
        for key in spec.src_keys() {
            encode_block_key(out, key);
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
}

fn decode_tree_transform_group_plan_f64(
    input: &mut CacheBytes<'_>,
) -> Result<TreeTransformGroupPlan<f64>, ()> {
    let spec_count = input.read_usize()?;
    let mut specs = Vec::with_capacity(spec_count);
    for _ in 0..spec_count {
        let group_key = decode_fusion_tree_group_key(input)?;
        let dst_count = input.read_usize()?;
        let mut dst_keys = Vec::with_capacity(dst_count);
        for _ in 0..dst_count {
            dst_keys.push(decode_block_key(input)?);
        }
        let src_count = input.read_usize()?;
        let mut src_keys = Vec::with_capacity(src_count);
        for _ in 0..src_count {
            src_keys.push(decode_block_key(input)?);
        }
        let coeff_count = input.read_usize()?;
        let mut coefficients = Vec::with_capacity(coeff_count);
        for _ in 0..coeff_count {
            coefficients.push(f64::from_bits(input.read_u64()?));
        }
        let mut spec =
            TreeTransformGroupBlockSpec::multi(group_key, dst_keys, src_keys, coefficients);
        if input.read_u8()? != 0 {
            let axis_count = input.read_usize()?;
            let mut axes = Vec::with_capacity(axis_count);
            for _ in 0..axis_count {
                axes.push(input.read_usize()?);
            }
            spec = spec.with_source_axes(axes);
        }
        specs.push(spec);
    }
    Ok(TreeTransformGroupPlan::new(specs))
}

fn encode_builtin_rule_key(out: &mut Vec<u8>, key: TreeTransformBuiltinRuleCacheKey) {
    out.push(match key {
        TreeTransformBuiltinRuleCacheKey::Z2 => 0,
        TreeTransformBuiltinRuleCacheKey::FermionParity => 1,
        TreeTransformBuiltinRuleCacheKey::U1 => 2,
        TreeTransformBuiltinRuleCacheKey::SU2 => 3,
    });
}

fn decode_builtin_rule_key(
    input: &mut CacheBytes<'_>,
) -> Result<TreeTransformBuiltinRuleCacheKey, ()> {
    match input.read_u8()? {
        0 => Ok(TreeTransformBuiltinRuleCacheKey::Z2),
        1 => Ok(TreeTransformBuiltinRuleCacheKey::FermionParity),
        2 => Ok(TreeTransformBuiltinRuleCacheKey::U1),
        3 => Ok(TreeTransformBuiltinRuleCacheKey::SU2),
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

fn encode_block_key(out: &mut Vec<u8>, key: &BlockKey) {
    match key {
        BlockKey::Dense => out.push(0),
        BlockKey::FusionTree(tree) => {
            out.push(1);
            encode_fusion_tree_block_key(out, tree);
        }
    }
}

fn decode_block_key(input: &mut CacheBytes<'_>) -> Result<BlockKey, ()> {
    match input.read_u8()? {
        0 => Ok(BlockKey::Dense),
        1 => Ok(BlockKey::FusionTree(decode_fusion_tree_block_key(input)?)),
        _ => Err(()),
    }
}

fn encode_fusion_tree_block_key(out: &mut Vec<u8>, key: &FusionTreeBlockKey) {
    encode_fusion_tree_key(out, key.codomain_tree());
    encode_fusion_tree_key(out, key.domain_tree());
}

fn decode_fusion_tree_block_key(input: &mut CacheBytes<'_>) -> Result<FusionTreeBlockKey, ()> {
    Ok(FusionTreeBlockKey::pair(
        decode_fusion_tree_key(input)?,
        decode_fusion_tree_key(input)?,
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

fn decode_fusion_tree_key(input: &mut CacheBytes<'_>) -> Result<FusionTreeKey, ()> {
    let uncoupled = decode_sector_vec(input)?;
    let coupled = if input.read_u8()? == 0 {
        None
    } else {
        Some(SectorId::new(input.read_usize()?))
    };
    let is_dual = decode_bool_vec(input)?;
    let innerlines = decode_sector_vec(input)?;
    let vertices = decode_sector_vec(input)?;
    FusionTreeKey::try_new_for_rule(
        &tenet_core::Z2FusionRule,
        uncoupled,
        coupled,
        is_dual,
        innerlines,
        vertices,
    )
    .map_err(|_| ())
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

    #[test]
    fn persistent_builtin_tree_plan_cache_round_trips_with_version_guard() {
        let group_key = FusionTreeGroupKey::from_sector_ids([1, 1], [], [false, false], []);
        let key = TreeTransformSectorPlanKey {
            rule: TreeTransformBuiltinRuleCacheKey::SU2,
            scope: TreeTransformPlanScope::TreePair,
            operation: TreeTransformOperation::braid([0, 1], [], [3, 5], []),
            source_groups: vec![TreeTransformSourceGroupKey {
                group_key: group_key.clone(),
                src_keys: vec![BlockKey::Dense],
            }],
        };
        let plan = Arc::new(TreeTransformGroupPlan::new(vec![
            TreeTransformGroupBlockSpec::multi(
                group_key,
                [BlockKey::Dense],
                [BlockKey::Dense],
                vec![1.25],
            )
            .with_source_axes([0, 1]),
        ]));
        let mut plans = FxHashMap::default();
        plans.insert(key.clone(), Arc::clone(&plan));

        let mut bytes = encode_builtin_tree_plan_cache(&plans);
        let decoded = decode_builtin_tree_plan_cache(&bytes).unwrap();
        assert_eq!(decoded, vec![(key, (*plan).clone())]);

        bytes[TREE_PLAN_CACHE_MAGIC.len()] = 99;
        assert!(decode_builtin_tree_plan_cache(&bytes).is_err());
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
    /// drives both replay and compile parallelism. `threads <= 1` is the
    /// untouched serial compile path.
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
        rule: &R,
        rule_key: &RuleKey,
        operation: TreeTransformOperation,
        src_structure: &BlockStructure,
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
            crate::tree_transform::plan::build_multiplicity_free_tree_pair_transform_group_plan_memoized(
            rule,
            rule_key,
            operation,
            src_structure,
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
        rule: &R,
        rule_key: &RuleKey,
        operation: TreeTransformOperation,
        src_structure: &BlockStructure,
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
            crate::tree_transform::plan::build_multiplicity_free_all_codomain_tree_transform_group_plan_memoized(
            rule,
            rule_key,
            operation,
            src_structure,
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
                build_tree_pair_transform_group_plan(rule, operation.clone(), src.structure())?;
            return Ok(Arc::new(plan.compile(dst, src)?));
        }
        if self.plans.contains_key(&plan_key) {
            self.stats.plan_hits += 1;
            self.touch_plan(&plan_key);
        } else {
            self.stats.plan_misses += 1;
            let plan = self.get_or_compile_global_tree_pair_plan(
                rule,
                &rule_key,
                operation.clone(),
                src.structure(),
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
                build_tree_pair_transform_group_plan(rule, operation.clone(), src_structure)?;
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
                rule,
                &rule_key,
                operation.clone(),
                src_structure,
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
        self.stats.plan_misses += 1;
        self.stats.structure_misses += 1;
        let plan = build_generic_tree_pair_transform_group_plan(rule, operation, src.structure())?;
        Ok(Arc::new(plan.compile(dst, src)?))
    }

    /// Structure-only generic sibling for the dynamic-rank (raw-slice) path —
    /// the top-level `tenet::Tensor` SU(3) `permute`/`braid`/`transpose` route.
    /// Same non-memoized rationale as [`Self::get_or_compile_tree_pair_generic`].
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
        self.stats.plan_misses += 1;
        self.stats.structure_misses += 1;
        let plan = build_generic_tree_pair_transform_group_plan(rule, operation, src_structure)?;
        Ok(Arc::new(
            plan.compile_shared_structures_with_storage_conjugation(
                Arc::clone(dst_structure),
                Arc::clone(src_structure),
                false,
            )?,
        ))
    }

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
        let plan_key = TreeTransformSectorPlanKey::new(
            rule_key.clone(),
            TreeTransformPlanScope::AllCodomain,
            operation.clone(),
            src.structure(),
        )?;
        if !self.policy.stores_entries() {
            self.stats.plan_misses += 1;
            self.stats.structure_misses += 1;
            let plan = crate::tree_transform::plan::build_multiplicity_free_all_codomain_tree_transform_group_plan(
                rule,
                operation.clone(),
                src.structure(),
            )?;
            return Ok(Arc::new(plan.compile(dst, src)?));
        }
        if self.plans.contains_key(&plan_key) {
            self.stats.plan_hits += 1;
            self.touch_plan(&plan_key);
        } else {
            self.stats.plan_misses += 1;
            let plan = self.get_or_compile_global_all_codomain_plan(
                rule,
                &rule_key,
                operation.clone(),
                src.structure(),
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
