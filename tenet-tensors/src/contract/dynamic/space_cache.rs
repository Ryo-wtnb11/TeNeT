use super::*;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct DynamicFusionSpaceCacheStats {
    hits: usize,
    fast_hits: usize,
    misses: usize,
}

impl DynamicFusionSpaceCacheStats {
    #[inline]
    pub(crate) fn hits(self) -> usize {
        self.hits
    }

    #[inline]
    pub(crate) fn fast_hits(self) -> usize {
        self.fast_hits
    }

    #[inline]
    pub(crate) fn misses(self) -> usize {
        self.misses
    }
}

#[derive(Clone, Debug)]
pub(crate) struct DynamicFusionSpaceCache<RuleKey, C = f64> {
    last_transformed_sources: Vec<DynamicFusionTransformedSourceLastEntry<RuleKey, C>>,
    fast_transformed_sources: FxHashMap<
        DynamicFusionTransformedSourceFastKey<RuleKey>,
        DynamicFusionTransformedSourceEntry<C>,
    >,
    transformed_sources: FxHashMap<
        DynamicFusionTransformedSourceSpaceKey<RuleKey>,
        DynamicFusionTransformedSourceEntry<C>,
    >,
    lru_order: VecDeque<DynamicFusionSpaceCacheEntryKey<RuleKey>>,
    last_core_dst: Option<DynamicFusionCoreDstLastEntry<RuleKey, C>>,
    fast_core_dsts: FxHashMap<DynamicFusionCoreDstFastKey<RuleKey>, DynamicFusionCoreDstEntry<C>>,
    core_dsts: FxHashMap<DynamicFusionCoreDstSpaceKey<RuleKey>, DynamicFusionCoreDstEntry<C>>,
    policy: OperationCachePolicy,
    stats: DynamicFusionSpaceCacheStats,
}

#[derive(Clone, Debug)]
pub(super) struct DynamicFusionTransformedSourceEntry<C = f64> {
    pub(super) space: Arc<DynamicFusionMapSpace>,
    pub(super) replay_structure: Arc<BlockStructure>,
    pub(super) transform_structure: Arc<TreeTransformStructure<C>>,
}

#[derive(Clone, Debug)]
pub(super) struct DynamicFusionCoreDstEntry<C = f64> {
    pub(super) space: Arc<DynamicFusionMapSpace>,
    pub(super) output_transform_structure: Arc<TreeTransformStructure<C>>,
}

impl<RuleKey, C> Default for DynamicFusionSpaceCache<RuleKey, C> {
    fn default() -> Self {
        Self {
            last_transformed_sources: Vec::new(),
            fast_transformed_sources: FxHashMap::default(),
            transformed_sources: FxHashMap::default(),
            lru_order: VecDeque::new(),
            last_core_dst: None,
            fast_core_dsts: FxHashMap::default(),
            core_dsts: FxHashMap::default(),
            policy: OperationCachePolicy::task_local_lru(DEFAULT_OPERATION_CACHE_ENTRIES),
            stats: DynamicFusionSpaceCacheStats::default(),
        }
    }
}

#[derive(Clone, Debug)]
struct DynamicFusionTransformedSourceLastEntry<RuleKey, C = f64> {
    key: Option<DynamicFusionTransformedSourceSpaceKey<RuleKey>>,
    rule: RuleKey,
    nout: usize,
    homspace: FusionTreeHomSpace,
    replay_structure: Arc<BlockStructure>,
    operation: TreeTransformOperation,
    source_conjugate: bool,
    entry: DynamicFusionTransformedSourceEntry<C>,
}

impl<RuleKey, C> DynamicFusionTransformedSourceLastEntry<RuleKey, C>
where
    RuleKey: Eq,
    C: Clone,
{
    fn matches(
        &self,
        rule: &RuleKey,
        nout: usize,
        homspace: &FusionTreeHomSpace,
        replay_structure: &Arc<BlockStructure>,
        operation: &TreeTransformOperation,
        source_conjugate: bool,
    ) -> bool {
        &self.rule == rule
            && self.nout == nout
            && self.homspace == *homspace
            && Arc::ptr_eq(&self.replay_structure, replay_structure)
            && &self.operation == operation
            && self.source_conjugate == source_conjugate
    }
}

#[derive(Clone, Debug)]
struct DynamicFusionCoreDstLastEntry<RuleKey, C = f64> {
    key: Option<DynamicFusionCoreDstSpaceKey<RuleKey>>,
    rule: RuleKey,
    lhs: DynamicFusionLastSpaceKey,
    rhs: DynamicFusionLastSpaceKey,
    core_axes: TensorContractSpecOwned,
    core_dst_open_lhs_rank: usize,
    core_dst_open_rhs_rank: usize,
    output_transform: TreeTransformOperation,
    output_dst: DynamicFusionLastSpaceKey,
    entry: DynamicFusionCoreDstEntry<C>,
}

impl<RuleKey, C> DynamicFusionCoreDstLastEntry<RuleKey, C>
where
    RuleKey: Eq,
{
    fn matches(
        &self,
        rule: &RuleKey,
        lhs: &DynamicFusionMapSpace,
        rhs: &DynamicFusionMapSpace,
        plan: &FusionContractPlan,
        output_dst: &DynamicFusionMapSpace,
    ) -> bool {
        &self.rule == rule
            && self.lhs.matches(lhs)
            && self.rhs.matches(rhs)
            && self.core_axes == *plan.core_axes()
            && self.core_dst_open_lhs_rank == plan.core_dst_open_lhs_rank()
            && self.core_dst_open_rhs_rank == plan.core_dst_open_rhs_rank()
            && self.output_transform == *plan.output_transform()
            && self.output_dst.matches(output_dst)
    }
}

#[derive(Clone, Debug)]
struct DynamicFusionLastSpaceKey {
    nout: usize,
    homspace: FusionTreeHomSpace,
    structure: Arc<BlockStructure>,
}

impl DynamicFusionLastSpaceKey {
    fn from_space(space: &DynamicFusionMapSpace) -> Self {
        Self {
            nout: space.nout(),
            homspace: space.homspace().clone(),
            structure: Arc::clone(space.structure()),
        }
    }

    fn matches(&self, space: &DynamicFusionMapSpace) -> bool {
        self.nout == space.nout()
            && self.homspace == *space.homspace()
            && Arc::ptr_eq(&self.structure, space.structure())
    }
}

impl<RuleKey, C> DynamicFusionSpaceCache<RuleKey, C>
where
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
    C: DenseBlockScalar,
{
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.transformed_sources.len() + self.core_dsts.len()
    }

    #[inline]
    pub(crate) fn stats(&self) -> DynamicFusionSpaceCacheStats {
        self.stats
    }

    #[inline]
    pub(crate) fn policy(&self) -> OperationCachePolicy {
        self.policy
    }

    pub(crate) fn set_policy(&mut self, policy: OperationCachePolicy) {
        self.policy = policy;
        self.clear_fast_entries();
        if !policy.stores_entries() {
            self.transformed_sources.clear();
            self.lru_order.clear();
            self.core_dsts.clear();
        } else if let Some(max_entries) = policy.max_entries() {
            self.rebuild_lru_order();
            self.enforce_lru_limit(max_entries);
        }
    }

    fn clear_fast_entries(&mut self) {
        self.last_transformed_sources.clear();
        self.fast_transformed_sources.clear();
        self.last_core_dst = None;
        self.fast_core_dsts.clear();
    }

    fn rebuild_lru_order(&mut self) {
        self.lru_order.clear();
        self.lru_order.extend(
            self.transformed_sources
                .keys()
                .cloned()
                .map(DynamicFusionSpaceCacheEntryKey::TransformedSource),
        );
        self.lru_order.extend(
            self.core_dsts
                .keys()
                .cloned()
                .map(DynamicFusionSpaceCacheEntryKey::CoreDst),
        );
    }

    fn remember_transformed_source(
        &mut self,
        entry: DynamicFusionTransformedSourceLastEntry<RuleKey, C>,
    ) {
        if !self.policy.stores_entries() {
            return;
        }
        const LAST_TRANSFORMED_SOURCE_LIMIT: usize = 4;
        if self.last_transformed_sources.len() == LAST_TRANSFORMED_SOURCE_LIMIT {
            self.last_transformed_sources.remove(0);
        }
        self.last_transformed_sources.push(entry);
    }

    fn touch_transformed_source(&mut self, key: &DynamicFusionTransformedSourceSpaceKey<RuleKey>) {
        if self.policy.max_entries().is_some() && self.transformed_sources.contains_key(key) {
            touch_lru_key(
                &mut self.lru_order,
                &DynamicFusionSpaceCacheEntryKey::TransformedSource(key.clone()),
            );
        }
    }

    fn insert_transformed_source(
        &mut self,
        key: DynamicFusionTransformedSourceSpaceKey<RuleKey>,
        fast_key: DynamicFusionTransformedSourceFastKey<RuleKey>,
        entry: DynamicFusionTransformedSourceEntry<C>,
    ) {
        if !self.policy.stores_entries() {
            return;
        }
        self.transformed_sources.insert(key.clone(), entry.clone());
        self.fast_transformed_sources.insert(fast_key, entry);
        if self.policy.max_entries().is_some() {
            self.touch_transformed_source(&key);
        }
        if let Some(max_entries) = self.policy.max_entries() {
            self.enforce_lru_limit(max_entries);
        }
    }

    fn touch_core_dst(&mut self, key: &DynamicFusionCoreDstSpaceKey<RuleKey>) {
        if self.policy.max_entries().is_some() && self.core_dsts.contains_key(key) {
            touch_lru_key(
                &mut self.lru_order,
                &DynamicFusionSpaceCacheEntryKey::CoreDst(key.clone()),
            );
        }
    }

    fn insert_core_dst(
        &mut self,
        key: DynamicFusionCoreDstSpaceKey<RuleKey>,
        fast_key: DynamicFusionCoreDstFastKey<RuleKey>,
        entry: DynamicFusionCoreDstEntry<C>,
    ) {
        if !self.policy.stores_entries() {
            return;
        }
        self.core_dsts.insert(key.clone(), entry.clone());
        self.fast_core_dsts.insert(fast_key, entry);
        if self.policy.max_entries().is_some() {
            self.touch_core_dst(&key);
        }
        if let Some(max_entries) = self.policy.max_entries() {
            self.enforce_lru_limit(max_entries);
        }
    }

    fn enforce_lru_limit(&mut self, max_entries: usize) {
        let mut evicted_transformed_source = false;
        let mut evicted_core_dst = false;
        while self.len() > max_entries {
            let Some(oldest) = self.lru_order.pop_front() else {
                break;
            };
            match oldest {
                DynamicFusionSpaceCacheEntryKey::TransformedSource(key) => {
                    evicted_transformed_source |= self.transformed_sources.remove(&key).is_some();
                }
                DynamicFusionSpaceCacheEntryKey::CoreDst(key) => {
                    evicted_core_dst |= self.core_dsts.remove(&key).is_some();
                }
            }
        }
        if evicted_transformed_source {
            self.last_transformed_sources.clear();
            self.fast_transformed_sources.clear();
        }
        if evicted_core_dst {
            self.last_core_dst = None;
            self.fast_core_dsts.clear();
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the cache lookup keeps source geometry, storage identity, transform, conjugation, and layout primer independently keyed"
    )]
    pub(super) fn get_or_compile_transformed_source<R, D, BT>(
        &mut self,
        tree_context: &mut TreeTransformExecutionContext<D, RuleKey, C, BT>,
        rule: &R,
        src_space: &DynamicFusionMapSpace,
        src_storage_structure: &Arc<BlockStructure>,
        operation: &TreeTransformOperation,
        source_conjugate: bool,
        layout_primer: LayoutKeyBuilder<R>,
    ) -> Result<DynamicFusionTransformedSourceEntry<C>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
        BT: TreeTransformBackend<D, C>,
    {
        let rule_key = rule.tree_transform_rule_cache_key();
        let nout = if source_conjugate {
            src_space.nin()
        } else {
            src_space.nout()
        };
        if self.policy.stores_entries() && !source_conjugate {
            let refresh_lru = self.policy.max_entries().is_some();
            let homspace = src_space.homspace();
            let replay_structure = src_storage_structure;
            let last_hit = self.last_transformed_sources.iter().find_map(|last| {
                if last.matches(
                    &rule_key,
                    nout,
                    homspace,
                    replay_structure,
                    operation,
                    source_conjugate,
                ) {
                    Some((
                        refresh_lru.then(|| last.key.clone()).flatten(),
                        last.entry.clone(),
                    ))
                } else {
                    None
                }
            });
            if let Some((key, entry)) = last_hit {
                self.stats.hits += 1;
                self.stats.fast_hits += 1;
                if let Some(key) = key.as_ref() {
                    self.touch_transformed_source(key);
                }
                return Ok(entry);
            }
        }
        let (homspace, replay_structure) = if source_conjugate {
            let adjoint = src_space.adjoint_view()?;
            (
                adjoint.homspace().clone(),
                std::sync::Arc::clone(adjoint.structure()),
            )
        } else {
            (
                src_space.homspace().clone(),
                std::sync::Arc::clone(src_storage_structure),
            )
        };
        if self.policy.stores_entries() && source_conjugate {
            let refresh_lru = self.policy.max_entries().is_some();
            let last_hit = self.last_transformed_sources.iter().find_map(|last| {
                if last.matches(
                    &rule_key,
                    nout,
                    &homspace,
                    &replay_structure,
                    operation,
                    source_conjugate,
                ) {
                    Some((
                        refresh_lru.then(|| last.key.clone()).flatten(),
                        last.entry.clone(),
                    ))
                } else {
                    None
                }
            });
            if let Some((key, entry)) = last_hit {
                self.stats.hits += 1;
                self.stats.fast_hits += 1;
                if let Some(key) = key.as_ref() {
                    self.touch_transformed_source(key);
                }
                return Ok(entry);
            }
        }
        if !self.policy.stores_entries() {
            self.stats.misses += 1;
            let space = if source_conjugate {
                src_space
                    .adjoint_view()?
                    .transformed_with_primer(rule, operation, layout_primer)?
            } else {
                src_space.transformed_with_primer(rule, operation, layout_primer)?
            };
            let dst_structure = Arc::clone(space.structure());
            let transform_structure = tree_context
                .get_or_compile_tree_pair_structure_with_storage_conjugation(
                    rule,
                    operation.clone(),
                    &dst_structure,
                    &replay_structure,
                    source_conjugate,
                )?;
            return Ok(DynamicFusionTransformedSourceEntry {
                space: Arc::new(space),
                replay_structure,
                transform_structure,
            });
        }

        let fast_key = DynamicFusionTransformedSourceFastKey {
            rule: rule_key.clone(),
            nout,
            homspace: homspace.clone(),
            replay_structure_id: replay_structure.content_id(),
            operation: operation.clone(),
            source_conjugate,
        };
        let lru_key = if self.policy.max_entries().is_some() {
            Some(DynamicFusionTransformedSourceSpaceKey {
                rule: rule_key.clone(),
                nout,
                homspace: homspace.clone(),
                structure: BlockStructureCacheKey::from_structure(&replay_structure)?,
                operation: operation.clone(),
                source_conjugate,
            })
        } else {
            None
        };
        if let Some(entry) = self.fast_transformed_sources.get(&fast_key) {
            let entry = entry.clone();
            self.stats.hits += 1;
            self.stats.fast_hits += 1;
            if let Some(key) = lru_key.as_ref() {
                self.touch_transformed_source(key);
            }
            self.remember_transformed_source(DynamicFusionTransformedSourceLastEntry {
                key: lru_key,
                rule: rule_key,
                nout,
                homspace,
                replay_structure,
                operation: operation.clone(),
                source_conjugate,
                entry: entry.clone(),
            });
            return Ok(entry);
        }
        let key = if let Some(key) = lru_key {
            key
        } else {
            DynamicFusionTransformedSourceSpaceKey {
                rule: rule_key.clone(),
                nout,
                homspace: homspace.clone(),
                structure: BlockStructureCacheKey::from_structure(&replay_structure)?,
                operation: operation.clone(),
                source_conjugate,
            }
        };
        if let Some(entry) = self.transformed_sources.get(&key) {
            let entry = entry.clone();
            self.stats.hits += 1;
            self.touch_transformed_source(&key);
            self.fast_transformed_sources
                .insert(fast_key, entry.clone());
            self.remember_transformed_source(DynamicFusionTransformedSourceLastEntry {
                key: Some(key.clone()),
                rule: rule_key,
                nout,
                homspace,
                replay_structure,
                operation: operation.clone(),
                source_conjugate,
                entry: entry.clone(),
            });
            return Ok(entry);
        }

        self.stats.misses += 1;
        let space = if source_conjugate {
            src_space
                .adjoint_view()?
                .transformed_with_primer(rule, operation, layout_primer)?
        } else {
            src_space.transformed_with_primer(rule, operation, layout_primer)?
        };
        let dst_structure = Arc::clone(space.structure());
        let transform_structure = tree_context
            .get_or_compile_tree_pair_structure_with_storage_conjugation(
                rule,
                operation.clone(),
                &dst_structure,
                &replay_structure,
                source_conjugate,
            )?;
        let entry = DynamicFusionTransformedSourceEntry {
            space: Arc::new(space),
            replay_structure,
            transform_structure,
        };
        let last_key = key.clone();
        self.insert_transformed_source(key, fast_key, entry.clone());
        self.remember_transformed_source(DynamicFusionTransformedSourceLastEntry {
            key: Some(last_key),
            rule: rule_key,
            nout,
            homspace,
            replay_structure: Arc::clone(&entry.replay_structure),
            operation: operation.clone(),
            source_conjugate,
            entry: entry.clone(),
        });
        Ok(entry)
    }

    pub(super) fn get_or_compile_transformed_source_oriented<R, D, BT>(
        &mut self,
        tree_context: &mut TreeTransformExecutionContext<D, RuleKey, C, BT>,
        rule: &R,
        source: &FusionOperandLayout<'_>,
        operation: &TreeTransformOperation,
        layout_primer: LayoutKeyBuilder<R>,
    ) -> Result<DynamicFusionTransformedSourceEntry<C>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
        BT: TreeTransformBackend<D, C>,
    {
        let rule_key = rule.tree_transform_rule_cache_key();
        let nout = source.nout();
        let homspace = source.homspace().clone();
        let replay_structure = Arc::clone(source.storage_space().structure());
        let source_conjugate = source.orientation() == FusionTreePairOrientation::Adjoint;
        let fast_key = DynamicFusionTransformedSourceFastKey {
            rule: rule_key.clone(),
            nout,
            homspace: homspace.clone(),
            replay_structure_id: replay_structure.content_id(),
            operation: operation.clone(),
            source_conjugate,
        };
        let key = DynamicFusionTransformedSourceSpaceKey {
            rule: rule_key.clone(),
            nout,
            homspace: homspace.clone(),
            structure: BlockStructureCacheKey::from_structure(&replay_structure)?,
            operation: operation.clone(),
            source_conjugate,
        };
        if self.policy.stores_entries() {
            if let Some(entry) = self.fast_transformed_sources.get(&fast_key).cloned() {
                self.stats.hits += 1;
                self.stats.fast_hits += 1;
                self.touch_transformed_source(&key);
                return Ok(entry);
            }
            if let Some(entry) = self.transformed_sources.get(&key).cloned() {
                self.stats.hits += 1;
                self.touch_transformed_source(&key);
                self.fast_transformed_sources
                    .insert(fast_key, entry.clone());
                return Ok(entry);
            }
        }

        self.stats.misses += 1;
        let space = source.transformed_space(rule, operation, layout_primer)?;
        let dst_structure = Arc::clone(space.structure());
        let logical_keys = source
            .adjoint_logical_keys()
            .expect("only adjoint sources use the oriented transform compiler");
        let transform_structure = tree_context.get_or_compile_tree_pair_structure_oriented(
            rule,
            operation,
            &dst_structure,
            logical_keys,
            || source.adjoint_storage_indices(),
            source.storage_space().structure(),
            source.orientation(),
            source.rank(),
            |axis| source.storage_axis(axis),
        )?;
        let entry = DynamicFusionTransformedSourceEntry {
            space: Arc::new(space),
            replay_structure,
            transform_structure,
        };
        if self.policy.stores_entries() {
            self.insert_transformed_source(key, fast_key, entry.clone());
        }
        Ok(entry)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the cache lookup keeps both source spaces, plan, output authority, and layout primer independently keyed"
    )]
    pub(super) fn get_or_compile_core_dst<R, D, BT>(
        &mut self,
        tree_context: &mut TreeTransformExecutionContext<D, RuleKey, C, BT>,
        rule: &R,
        lhs: &DynamicFusionMapSpace,
        rhs: &DynamicFusionMapSpace,
        plan: &FusionContractPlan,
        output_dst: &DynamicFusionMapSpace,
        layout_primer: LayoutKeyBuilder<R>,
    ) -> Result<DynamicFusionCoreDstEntry<C>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
        BT: TreeTransformBackend<D, C>,
    {
        let rule_key = rule.tree_transform_rule_cache_key();
        if !self.policy.stores_entries() {
            self.stats.misses += 1;
            let space =
                DynamicFusionMapSpace::core_dst_with_primer(rule, lhs, rhs, plan, layout_primer)?;
            let dst_structure = Arc::clone(output_dst.structure());
            let src_structure = Arc::clone(space.structure());
            let output_transform_structure = tree_context
                .get_or_compile_tree_pair_structure_with_storage_conjugation(
                    rule,
                    plan.output_transform().clone(),
                    &dst_structure,
                    &src_structure,
                    false,
                )?;
            return Ok(DynamicFusionCoreDstEntry {
                space: Arc::new(space),
                output_transform_structure,
            });
        }
        if let Some(last) = &self.last_core_dst {
            if last.matches(&rule_key, lhs, rhs, plan, output_dst) {
                let key = self
                    .policy
                    .max_entries()
                    .is_some()
                    .then(|| last.key.clone())
                    .flatten();
                let entry = last.entry.clone();
                self.stats.hits += 1;
                self.stats.fast_hits += 1;
                if let Some(key) = key.as_ref() {
                    self.touch_core_dst(key);
                }
                return Ok(entry);
            }
        }
        let fast_key = DynamicFusionCoreDstFastKey {
            rule: rule_key.clone(),
            lhs: DynamicFusionFastSpaceKey::from_space(lhs),
            rhs: DynamicFusionFastSpaceKey::from_space(rhs),
            core_axes: plan.core_axes().clone(),
            core_dst_open_lhs_rank: plan.core_dst_open_lhs_rank(),
            core_dst_open_rhs_rank: plan.core_dst_open_rhs_rank(),
            output_transform: plan.output_transform().clone(),
            output_dst: DynamicFusionFastSpaceKey::from_space(output_dst),
        };
        let lru_key = if self.policy.max_entries().is_some() {
            Some(DynamicFusionCoreDstSpaceKey {
                rule: rule_key.clone(),
                lhs: DynamicFusionSpaceKey::from_space(lhs)?,
                rhs: DynamicFusionSpaceKey::from_space(rhs)?,
                core_axes: plan.core_axes().clone(),
                core_dst_open_lhs_rank: plan.core_dst_open_lhs_rank(),
                core_dst_open_rhs_rank: plan.core_dst_open_rhs_rank(),
                output_transform: plan.output_transform().clone(),
                output_dst: DynamicFusionSpaceKey::from_space(output_dst)?,
            })
        } else {
            None
        };
        if let Some(entry) = self.fast_core_dsts.get(&fast_key) {
            let entry = entry.clone();
            self.stats.hits += 1;
            self.stats.fast_hits += 1;
            if let Some(key) = lru_key.as_ref() {
                self.touch_core_dst(key);
            }
            self.last_core_dst = Some(DynamicFusionCoreDstLastEntry {
                key: lru_key,
                rule: rule_key,
                lhs: DynamicFusionLastSpaceKey::from_space(lhs),
                rhs: DynamicFusionLastSpaceKey::from_space(rhs),
                core_axes: plan.core_axes().clone(),
                core_dst_open_lhs_rank: plan.core_dst_open_lhs_rank(),
                core_dst_open_rhs_rank: plan.core_dst_open_rhs_rank(),
                output_transform: plan.output_transform().clone(),
                output_dst: DynamicFusionLastSpaceKey::from_space(output_dst),
                entry: entry.clone(),
            });
            return Ok(entry);
        }
        let key = if let Some(key) = lru_key {
            key
        } else {
            DynamicFusionCoreDstSpaceKey {
                rule: rule_key.clone(),
                lhs: DynamicFusionSpaceKey::from_space(lhs)?,
                rhs: DynamicFusionSpaceKey::from_space(rhs)?,
                core_axes: plan.core_axes().clone(),
                core_dst_open_lhs_rank: plan.core_dst_open_lhs_rank(),
                core_dst_open_rhs_rank: plan.core_dst_open_rhs_rank(),
                output_transform: plan.output_transform().clone(),
                output_dst: DynamicFusionSpaceKey::from_space(output_dst)?,
            }
        };
        if let Some(entry) = self.core_dsts.get(&key) {
            let entry = entry.clone();
            self.stats.hits += 1;
            self.touch_core_dst(&key);
            self.fast_core_dsts.insert(fast_key, entry.clone());
            self.last_core_dst = Some(DynamicFusionCoreDstLastEntry {
                key: Some(key.clone()),
                rule: rule_key,
                lhs: DynamicFusionLastSpaceKey::from_space(lhs),
                rhs: DynamicFusionLastSpaceKey::from_space(rhs),
                core_axes: plan.core_axes().clone(),
                core_dst_open_lhs_rank: plan.core_dst_open_lhs_rank(),
                core_dst_open_rhs_rank: plan.core_dst_open_rhs_rank(),
                output_transform: plan.output_transform().clone(),
                output_dst: DynamicFusionLastSpaceKey::from_space(output_dst),
                entry: entry.clone(),
            });
            return Ok(entry);
        }

        self.stats.misses += 1;
        let space =
            DynamicFusionMapSpace::core_dst_with_primer(rule, lhs, rhs, plan, layout_primer)?;
        let dst_structure = Arc::clone(output_dst.structure());
        let src_structure = Arc::clone(space.structure());
        let output_transform_structure = tree_context
            .get_or_compile_tree_pair_structure_with_storage_conjugation(
                rule,
                plan.output_transform().clone(),
                &dst_structure,
                &src_structure,
                false,
            )?;
        let entry = DynamicFusionCoreDstEntry {
            space: Arc::new(space),
            output_transform_structure,
        };
        let last_key = key.clone();
        self.insert_core_dst(key, fast_key, entry.clone());
        self.last_core_dst = Some(DynamicFusionCoreDstLastEntry {
            key: Some(last_key),
            rule: rule_key,
            lhs: DynamicFusionLastSpaceKey::from_space(lhs),
            rhs: DynamicFusionLastSpaceKey::from_space(rhs),
            core_axes: plan.core_axes().clone(),
            core_dst_open_lhs_rank: plan.core_dst_open_lhs_rank(),
            core_dst_open_rhs_rank: plan.core_dst_open_rhs_rank(),
            output_transform: plan.output_transform().clone(),
            output_dst: DynamicFusionLastSpaceKey::from_space(output_dst),
            entry: entry.clone(),
        });
        Ok(entry)
    }
}

#[expect(
    clippy::large_enum_variant,
    reason = "LRU order entries hold keys by value: boxing the core-destination key would allocate on every touch of a bounded cache"
)]
#[derive(Clone, Debug, Eq, PartialEq)]
enum DynamicFusionSpaceCacheEntryKey<RuleKey> {
    TransformedSource(DynamicFusionTransformedSourceSpaceKey<RuleKey>),
    CoreDst(DynamicFusionCoreDstSpaceKey<RuleKey>),
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(super) struct DynamicFusionTransformedSourceFastKey<RuleKey> {
    pub(super) rule: RuleKey,
    pub(super) nout: usize,
    pub(super) homspace: FusionTreeHomSpace,
    pub(super) replay_structure_id: usize,
    pub(super) operation: TreeTransformOperation,
    pub(super) source_conjugate: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct DynamicFusionTransformedSourceSpaceKey<RuleKey> {
    rule: RuleKey,
    nout: usize,
    homspace: FusionTreeHomSpace,
    structure: BlockStructureCacheKey,
    operation: TreeTransformOperation,
    source_conjugate: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct DynamicFusionCoreDstFastKey<RuleKey> {
    rule: RuleKey,
    lhs: DynamicFusionFastSpaceKey,
    rhs: DynamicFusionFastSpaceKey,
    core_axes: TensorContractSpecOwned,
    core_dst_open_lhs_rank: usize,
    core_dst_open_rhs_rank: usize,
    output_transform: TreeTransformOperation,
    output_dst: DynamicFusionFastSpaceKey,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct DynamicFusionCoreDstSpaceKey<RuleKey> {
    rule: RuleKey,
    lhs: DynamicFusionSpaceKey,
    rhs: DynamicFusionSpaceKey,
    core_axes: TensorContractSpecOwned,
    core_dst_open_lhs_rank: usize,
    core_dst_open_rhs_rank: usize,
    output_transform: TreeTransformOperation,
    output_dst: DynamicFusionSpaceKey,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(super) struct DynamicFusionFastSpaceKey {
    pub(super) nout: usize,
    pub(super) homspace: FusionTreeHomSpace,
    pub(super) structure_id: usize,
}

impl DynamicFusionFastSpaceKey {
    fn from_space(space: &DynamicFusionMapSpace) -> Self {
        Self {
            nout: space.nout(),
            homspace: space.homspace().clone(),
            structure_id: space.structure().content_id(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct DynamicFusionSpaceKey {
    nout: usize,
    homspace: FusionTreeHomSpace,
    structure: BlockStructureCacheKey,
}

impl DynamicFusionSpaceKey {
    fn from_space(space: &DynamicFusionMapSpace) -> Result<Self, OperationError> {
        Ok(Self {
            nout: space.nout(),
            homspace: space.homspace().clone(),
            structure: BlockStructureCacheKey::from_structure(space.structure())?,
        })
    }
}
