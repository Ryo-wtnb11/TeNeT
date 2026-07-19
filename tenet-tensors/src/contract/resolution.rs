//! Unified contraction resolution: one cache maps
//! `(rule, dst/lhs/rhs spaces, axes)` to the resolved execution artifact —
//! route and plan together.
//!
//! TensorKit keeps exactly one cache entry per (spaces, permutation)
//! (`treepermuter` and friends); its route decision is compile-time dispatch
//! plus a cheap uncached cost heuristic and is never memoized. The previous
//! three TeNeT caches (route / core fusion-block plan / explicit plan)
//! all shared this key with different payloads — this module restores the
//! one-entry granularity. Prepared contraction handles wrap the same
//! [`Resolution`] value, so the facade and the plan-once API share one
//! resolution machinery.

use std::collections::VecDeque;

use rustc_hash::FxHashMap;
use std::hash::Hash;
use std::sync::{Arc, RwLock};

use tenet_core::{BlockStructure, FusionTreeHomSpace, HomSpaceId, MultiplicityFreeRigidSymbols};

use super::structure::TensorContractStructure;
use crate::cache::{typed_global_map, BlockStructureCacheKey, OperationCachePolicy};
use crate::OperationError;
use tenet_operations::axis::{OutputAxisOrder, TensorContractSpec, TensorContractSpecOwned};
use tenet_operations::fusion_replay::FusionBlockContractPlan;

use super::dynamic_space::DynamicFusionMapSpace;
use super::fusion::{external_axis_is_dual, FusionContractPlan};
use super::fusion_block::{
    compile_fusion_block_contract_plan_prelowered, compile_fusion_block_contract_plan_validated,
    is_core_form_fusion_block_contract, validate_fusion_contract_rule,
};
use super::structure::TensorContractAxisPlan;

type GlobalContractionResolutionMap<RuleKey> = RwLock<FxHashMap<FullKey<RuleKey>, Resolution>>;

fn global_contraction_resolutions<RuleKey>() -> Arc<GlobalContractionResolutionMap<RuleKey>>
where
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
{
    typed_global_map()
}

/// Resolved execution artifact for one contraction key: the route decision
/// and its compiled plan are one value, never cached separately.
#[derive(Clone, Debug)]
pub(crate) enum Resolution {
    /// Coupled-sector direct GEMM (TensorKit `mul!` shape).
    Core(Arc<FusionBlockContractPlan>),
    /// Source/output tree transforms around a core core
    /// (TensorKit `@tensor` shape).
    DynamicTree(Arc<FusionContractPlan>),
    /// Dense one-shot structure for conjugated operands (TeNeT optimization
    /// over the faithful transform-then-contract path).
    Structure(Arc<TensorContractStructure<f64>>),
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ContractionResolutionStats {
    pub(crate) hits: usize,
    pub(crate) fast_hits: usize,
    pub(crate) misses: usize,
}

/// Structural full key: reachable identically from typed and dynamic callers.
/// `scope` separates ordinary resolution, prelowered storage-dependent
/// resolution, and the dynamic route's internal scratch plans.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct FullKey<RuleKey> {
    rule: RuleKey,
    dst: FullSpaceKey,
    lhs: FullSpaceKey,
    rhs: FullSpaceKey,
    axes: TensorContractSpecOwned,
    scope: FullResolutionScope,
    access: OperandAccess,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
enum FullResolutionScope {
    Ordinary,
    Prelowered {
        lhs_storage: FullSpaceKey,
        rhs_storage: FullSpaceKey,
    },
    CoreOnly,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
struct OperandAccess {
    lhs: OperandAccessMode,
    rhs: OperandAccessMode,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
enum OperandAccessMode {
    #[default]
    Direct,
    AdjointParent,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct FullSpaceKey {
    nout: usize,
    homspace: FusionTreeHomSpace,
    structure: BlockStructureCacheKey,
}

impl FullSpaceKey {
    fn from_space(space: &DynamicFusionMapSpace) -> Result<Self, OperationError> {
        Ok(Self {
            nout: space.nout(),
            homspace: space.homspace().clone(),
            structure: BlockStructureCacheKey::from_structure(space.structure())?,
        })
    }
}

/// Content-identity fast key: the cheap-hash precursor to [`FullKey`], serving
/// many distinct keys without the full key's deep hom-space hash, so loops
/// cycling through several fixed contractions (an iPEPS unit cell, energy
/// environments) stay O(1) on the key hash even when they alternate. Keyed on
/// the same interned semantic identities `FullKey` resolves to — the hom space's
/// [`HomSpaceId`] (O(1) prehash) and the block structure's `content_id` —
/// instead of the deep clone-and-hash of the whole `FusionTreeHomSpace`.
///
/// Why-not (raw `Arc::as_ptr` keys, the previous form): a pointer key is
/// principally unsound (ABA — a freed operand Arc's address can be reused by an
/// unrelated live structure) and hits only when the *same* Arc recurs. Content
/// keys are sound and additionally hit when identical content arrives in a fresh
/// Arc — a hit the pointer key missed. That extra hit is semantically safe for
/// every [`Resolution`] payload: `Core`/`Structure` carry their own pinned
/// operand structures and replay re-validates them against the live operands
/// (`validate_structure_identity`), and `DynamicTree` plans are pure functions
/// of (rank, axes, conj) and content-independent (the χ1-vs-χ3 regression test
/// pins that invariant). This is exactly what a `FullKey` hit already does —
/// `FullKey` and `FastKey` now share one content-equivalence class, so a
/// `FastKey` hit is always a would-be `FullKey` hit, never a divergent one.
///
/// Why-not (pin the operand Arcs instead, à la [`LastSpace`]): pinning keeps
/// dead structures alive only to keep a pointer key valid; re-keying on content
/// removes the unsoundness at the root without extending any lifetime.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct FastKey<RuleKey> {
    rule: RuleKey,
    dst: FastSpaceKey,
    lhs: FastSpaceKey,
    rhs: FastSpaceKey,
    axes: TensorContractSpecOwned,
    scope: FastResolutionScope,
    access: OperandAccess,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
enum FastResolutionScope {
    Ordinary,
    Prelowered {
        lhs_storage: FastSpaceKey,
        rhs_storage: FastSpaceKey,
    },
    CoreOnly,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct FastSpaceKey {
    nout: usize,
    homspace_id: HomSpaceId,
    structure_id: usize,
}

impl FastSpaceKey {
    fn from_space(space: &DynamicFusionMapSpace) -> Self {
        Self {
            nout: space.nout(),
            homspace_id: space.homspace().id(),
            structure_id: space.structure().content_id(),
        }
    }
}

#[derive(Clone, Debug)]
struct LastEntry<RuleKey> {
    rule: RuleKey,
    dst: LastSpace,
    lhs: LastSpace,
    rhs: LastSpace,
    axes: RawAxes,
    scope: LastResolutionScope,
    access: OperandAccess,
    full_key: Option<FullKey<RuleKey>>,
    resolution: Resolution,
}

#[derive(Clone, Debug)]
enum LastResolutionScope {
    Ordinary,
    Prelowered {
        lhs_storage: LastSpace,
        rhs_storage: LastSpace,
    },
    CoreOnly,
}

#[derive(Clone, Copy, Debug)]
enum LiveResolutionScope<'a> {
    Ordinary,
    Prelowered {
        lhs_storage: &'a DynamicFusionMapSpace,
        rhs_storage: &'a DynamicFusionMapSpace,
    },
    CoreOnly,
}

#[derive(Clone, Debug)]
struct LastSpace {
    nout: usize,
    homspace: Arc<FusionTreeHomSpace>,
    structure: Arc<BlockStructure>,
}

impl LastSpace {
    fn from_space(space: &DynamicFusionMapSpace) -> Self {
        Self {
            nout: space.nout(),
            homspace: Arc::clone(space.homspace_arc()),
            structure: Arc::clone(space.structure()),
        }
    }

    fn matches(&self, space: &DynamicFusionMapSpace) -> bool {
        self.nout == space.nout()
            && Arc::ptr_eq(&self.structure, space.structure())
            && (Arc::ptr_eq(&self.homspace, space.homspace_arc())
                || *self.homspace == *space.homspace())
    }
}

impl LastResolutionScope {
    fn matches(&self, live: LiveResolutionScope<'_>) -> bool {
        match (self, live) {
            (Self::Ordinary, LiveResolutionScope::Ordinary)
            | (Self::CoreOnly, LiveResolutionScope::CoreOnly) => true,
            (
                Self::Prelowered {
                    lhs_storage,
                    rhs_storage,
                },
                LiveResolutionScope::Prelowered {
                    lhs_storage: live_lhs,
                    rhs_storage: live_rhs,
                },
            ) => lhs_storage.matches(live_lhs) && rhs_storage.matches(live_rhs),
            _ => false,
        }
    }
}

impl LiveResolutionScope<'_> {
    fn full(self) -> Result<FullResolutionScope, OperationError> {
        match self {
            Self::Ordinary => Ok(FullResolutionScope::Ordinary),
            Self::Prelowered {
                lhs_storage,
                rhs_storage,
            } => Ok(FullResolutionScope::Prelowered {
                lhs_storage: FullSpaceKey::from_space(lhs_storage)?,
                rhs_storage: FullSpaceKey::from_space(rhs_storage)?,
            }),
            Self::CoreOnly => Ok(FullResolutionScope::CoreOnly),
        }
    }

    fn fast(self) -> FastResolutionScope {
        match self {
            Self::Ordinary => FastResolutionScope::Ordinary,
            Self::Prelowered {
                lhs_storage,
                rhs_storage,
            } => FastResolutionScope::Prelowered {
                lhs_storage: FastSpaceKey::from_space(lhs_storage),
                rhs_storage: FastSpaceKey::from_space(rhs_storage),
            },
            Self::CoreOnly => FastResolutionScope::CoreOnly,
        }
    }

    fn pinned(self) -> LastResolutionScope {
        match self {
            Self::Ordinary => LastResolutionScope::Ordinary,
            Self::Prelowered {
                lhs_storage,
                rhs_storage,
            } => LastResolutionScope::Prelowered {
                lhs_storage: LastSpace::from_space(lhs_storage),
                rhs_storage: LastSpace::from_space(rhs_storage),
            },
            Self::CoreOnly => LastResolutionScope::CoreOnly,
        }
    }
}

/// Raw axes copy for last-entry comparison without recompiling the axis plan.
#[derive(Clone, Debug, Eq, PartialEq)]
struct RawAxes {
    lhs_contracting: Vec<usize>,
    rhs_contracting: Vec<usize>,
    output_identity: bool,
    output_axes: Vec<usize>,
    lhs_conjugate: bool,
    rhs_conjugate: bool,
}

impl RawAxes {
    fn from_axes(axes: TensorContractSpec<'_>) -> Self {
        let (output_identity, output_axes) = match axes.output_permutation() {
            OutputAxisOrder::Identity => (true, Vec::new()),
            OutputAxisOrder::Axes(list) => (false, list.to_vec()),
        };
        Self {
            lhs_contracting: axes.lhs_contracting_axes().to_vec(),
            rhs_contracting: axes.rhs_contracting_axes().to_vec(),
            output_identity,
            output_axes,
            lhs_conjugate: axes.lhs_conjugate(),
            rhs_conjugate: axes.rhs_conjugate(),
        }
    }

    fn matches(&self, axes: TensorContractSpec<'_>) -> bool {
        let output_matches = match axes.output_permutation() {
            OutputAxisOrder::Identity => self.output_identity,
            OutputAxisOrder::Axes(list) => !self.output_identity && self.output_axes == list,
        };
        output_matches
            && self.lhs_contracting == axes.lhs_contracting_axes()
            && self.rhs_contracting == axes.rhs_contracting_axes()
            && self.lhs_conjugate == axes.lhs_conjugate()
            && self.rhs_conjugate == axes.rhs_conjugate()
    }
}

/// Recently used entries kept for pointer-compared lookups. A dynamic
/// contraction interleaves the facade key with several distinct internal
/// scratch keys per replay, so a single slot would thrash on every call.
const LAST_RING_CAPACITY: usize = 16;

/// Cap on the content-keyed fast map. The fast map is a pure accelerator —
/// every entry is reconstructible from `resolved`, so bounding it only costs a
/// re-promotion (a `FullKey` hash) on the next miss, never correctness. This
/// bounds fast-map growth even under the unbounded `TaskLocal` policy (the P3
/// hardening: a long-lived process no longer grows it monotonically). Under an
/// LRU policy the fast cap tracks `max_entries`, so fast never falls behind
/// `resolved` and no working-set key needlessly pays the deep `FullKey` hash.
const FAST_MAP_CAPACITY: usize = 4096;

#[derive(Clone, Debug)]
pub(crate) struct ContractionResolutionCache<RuleKey> {
    last: Vec<LastEntry<RuleKey>>,
    fast: FxHashMap<FastKey<RuleKey>, Resolution>,
    resolved: FxHashMap<FullKey<RuleKey>, Resolution>,
    lru_order: VecDeque<FullKey<RuleKey>>,
    policy: OperationCachePolicy,
    stats: ContractionResolutionStats,
}

impl<RuleKey> Default for ContractionResolutionCache<RuleKey> {
    fn default() -> Self {
        Self {
            last: Vec::new(),
            fast: FxHashMap::default(),
            resolved: FxHashMap::default(),
            lru_order: VecDeque::new(),
            policy: OperationCachePolicy::default(),
            stats: ContractionResolutionStats::default(),
        }
    }
}

impl<RuleKey> ContractionResolutionCache<RuleKey>
where
    RuleKey: Clone + Eq + Hash,
{
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.resolved.len()
    }

    #[inline]
    pub(crate) fn stats(&self) -> ContractionResolutionStats {
        self.stats
    }

    pub(crate) fn set_policy(&mut self, policy: OperationCachePolicy) {
        self.policy = policy;
        self.last.clear();
        self.fast.clear();
        if !policy.stores_entries() {
            self.resolved.clear();
            self.lru_order.clear();
        } else if let Some(max_entries) = policy.max_entries() {
            crate::cache::rebuild_lru_order_from_keys(&self.resolved, &mut self.lru_order);
            self.enforce_lru_limit(max_entries);
        }
    }

    fn enforce_lru_limit(&mut self, max_entries: usize) {
        let mut evicted = false;
        while self.resolved.len() > max_entries {
            let Some(oldest) = self.lru_order.pop_front() else {
                break;
            };
            evicted |= self.resolved.remove(&oldest).is_some();
        }
        if evicted {
            self.fast.clear();
            self.last.clear();
        }
    }

    fn fast_insert(&mut self, key: FastKey<RuleKey>, resolution: Resolution) {
        // ponytail: nuke-on-full, not per-entry LRU — keeps the hot fast-hit
        // path free of reorder bookkeeping (the `last` ring already absorbs the
        // hottest keys, and a lost fast entry only falls through to `resolved`).
        // Upgrade to keyed LRU only if fast-map churn shows up in a profile.
        let cap = self.policy.max_entries().unwrap_or(FAST_MAP_CAPACITY);
        if self.fast.len() >= cap && !self.fast.contains_key(&key) {
            self.fast.clear();
        }
        self.fast.insert(key, resolution);
    }

    fn touch(&mut self, key: &FullKey<RuleKey>) {
        if self.policy.max_entries().is_some() {
            crate::cache::touch_lru_key(&mut self.lru_order, key);
        }
    }

    /// Resolve route and plan in one lookup. On a cold key the core
    /// plan compiled during the route test is stored, never discarded;
    /// non-core keys resolve through `compile_dynamic` (which has the
    /// caller's typed context) or, for conjugated axes, `compile_structure`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn get_or_resolve<R>(
        &mut self,
        rule: &R,
        dst: &DynamicFusionMapSpace,
        lhs: &DynamicFusionMapSpace,
        rhs: &DynamicFusionMapSpace,
        axes: TensorContractSpec<'_>,
        compile_structure: impl FnOnce() -> Result<
            Option<Arc<TensorContractStructure<f64>>>,
            OperationError,
        >,
        compile_dynamic: impl FnOnce() -> Result<Arc<FusionContractPlan>, OperationError>,
    ) -> Result<Resolution, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>
            + crate::TreeTransformRuleCacheKey<Key = RuleKey>,
        RuleKey: 'static + Send + Sync,
    {
        self.get_or_resolve_with(
            rule,
            dst,
            lhs,
            rhs,
            axes,
            LiveResolutionScope::Ordinary,
            OperandAccess::default(),
            || {
                resolve(
                    rule,
                    dst,
                    lhs,
                    rhs,
                    axes,
                    compile_structure,
                    compile_dynamic,
                )
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn get_or_resolve_prelowered<R>(
        &mut self,
        rule: &R,
        dst: &DynamicFusionMapSpace,
        lhs_logical: &DynamicFusionMapSpace,
        lhs_storage: &DynamicFusionMapSpace,
        rhs_logical: &DynamicFusionMapSpace,
        rhs_storage: &DynamicFusionMapSpace,
        axes: TensorContractSpec<'_>,
        compile_structure: impl FnOnce() -> Result<
            Option<Arc<TensorContractStructure<f64>>>,
            OperationError,
        >,
        compile_dynamic: impl FnOnce() -> Result<Arc<FusionContractPlan>, OperationError>,
    ) -> Result<Resolution, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>
            + crate::TreeTransformRuleCacheKey<Key = RuleKey>,
        RuleKey: 'static + Send + Sync,
    {
        // Why not rely on the public context's preflight: this cache owns the
        // storage-dependent identity and must reject foreign storage before a
        // last/fast/global hit if another internal caller is added later.
        lhs_storage.validate_rule(rule)?;
        rhs_storage.validate_rule(rule)?;
        let access = OperandAccess {
            lhs: if axes.lhs_conjugate() {
                OperandAccessMode::AdjointParent
            } else {
                OperandAccessMode::Direct
            },
            rhs: if axes.rhs_conjugate() {
                OperandAccessMode::AdjointParent
            } else {
                OperandAccessMode::Direct
            },
        };
        self.get_or_resolve_with(
            rule,
            dst,
            lhs_logical,
            rhs_logical,
            axes,
            LiveResolutionScope::Prelowered {
                lhs_storage,
                rhs_storage,
            },
            access,
            || {
                let logical_axes = TensorContractSpec::new(
                    axes.lhs_contracting_axes(),
                    axes.rhs_contracting_axes(),
                    axes.output_permutation(),
                );
                if is_core_form_fusion_block_contract(
                    rule,
                    dst,
                    lhs_logical,
                    rhs_logical,
                    logical_axes,
                )? && !rhs_contract_requires_twist(rule, rhs_logical, logical_axes)?
                {
                    let plan = compile_fusion_block_contract_plan_prelowered(
                        rule,
                        dst,
                        lhs_logical,
                        lhs_storage,
                        rhs_logical,
                        rhs_storage,
                        logical_axes,
                        if axes.lhs_conjugate() {
                            tenet_operations::fusion_replay::MatrixOp::Adjoint
                        } else {
                            tenet_operations::fusion_replay::MatrixOp::Identity
                        },
                        if axes.rhs_conjugate() {
                            tenet_operations::fusion_replay::MatrixOp::Adjoint
                        } else {
                            tenet_operations::fusion_replay::MatrixOp::Identity
                        },
                    )?;
                    if plan.is_fully_direct() {
                        return Ok(Resolution::Core(Arc::new(plan)));
                    }
                }
                if let Some(structure) = compile_structure()? {
                    return Ok(Resolution::Structure(structure));
                }
                Ok(Resolution::DynamicTree(compile_dynamic()?))
            },
        )
    }

    /// Core-only entry for the dynamic route's internal scratch
    /// contractions: operands are already materialized in coupled scratch
    /// (twists applied), so the route test and twist gate do not apply.
    pub(crate) fn get_or_compile_core_plan<R>(
        &mut self,
        rule: &R,
        dst: &DynamicFusionMapSpace,
        lhs: &DynamicFusionMapSpace,
        rhs: &DynamicFusionMapSpace,
        axes: TensorContractSpec<'_>,
    ) -> Result<Arc<FusionBlockContractPlan>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>
            + crate::TreeTransformRuleCacheKey<Key = RuleKey>,
        RuleKey: 'static + Send + Sync,
    {
        let resolution = self.get_or_resolve_with(
            rule,
            dst,
            lhs,
            rhs,
            axes,
            LiveResolutionScope::CoreOnly,
            OperandAccess::default(),
            || {
                compile_fusion_block_contract_plan_validated(rule, dst, lhs, rhs, axes)
                    .map(|plan| Resolution::Core(Arc::new(plan)))
            },
        )?;
        match resolution {
            Resolution::Core(plan) => Ok(plan),
            _ => Err(OperationError::UnsupportedTensorContractScope {
                message: "internal scratch contraction resolved to a non-core plan",
            }),
        }
    }

    /// Shared lookup skeleton: fast paths, then `resolve_cold` on a miss.
    fn get_or_resolve_with<R>(
        &mut self,
        rule: &R,
        dst: &DynamicFusionMapSpace,
        lhs: &DynamicFusionMapSpace,
        rhs: &DynamicFusionMapSpace,
        axes: TensorContractSpec<'_>,
        scope: LiveResolutionScope<'_>,
        access: OperandAccess,
        resolve_cold: impl FnOnce() -> Result<Resolution, OperationError>,
    ) -> Result<Resolution, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>
            + crate::TreeTransformRuleCacheKey<Key = RuleKey>,
        RuleKey: 'static + Send + Sync,
    {
        validate_fusion_contract_rule(rule, dst, lhs, rhs)?;
        let rule_key = rule.tree_transform_rule_cache_key();
        if self.policy.stores_entries() {
            let position = self.last.iter().position(|last| {
                last.rule == rule_key
                    && last.scope.matches(scope)
                    && last.access == access
                    && last.dst.matches(dst)
                    && last.lhs.matches(lhs)
                    && last.rhs.matches(rhs)
                    && last.axes.matches(axes)
            });
            if let Some(index) = position {
                self.stats.hits += 1;
                self.stats.fast_hits += 1;
                if index != 0 {
                    let entry = self.last.remove(index);
                    self.last.insert(0, entry);
                }
                let resolution = self.last[0].resolution.clone();
                // Clone the deep structural key only when an LRU limit
                // actually needs the touch; the clone dwarfs the whole
                // hit path otherwise.
                let touch_key = if self.policy.max_entries().is_some() {
                    self.last[0].full_key.clone()
                } else {
                    None
                };
                if let Some(key) = touch_key {
                    self.touch(&key);
                }
                return Ok(resolution);
            }
        }

        let axis_plan = TensorContractAxisPlan::compile(lhs.rank(), rhs.rank(), dst.rank(), axes)?;
        let axes_key = TensorContractSpecOwned::new_with_conjugation(
            axis_plan.lhs_contracting_axes,
            axis_plan.rhs_contracting_axes,
            axis_plan.output_axes,
            axis_plan.lhs_conjugate,
            axis_plan.rhs_conjugate,
        );

        if self.policy.stores_entries() {
            let fast_key = FastKey {
                rule: rule_key.clone(),
                dst: FastSpaceKey::from_space(dst),
                lhs: FastSpaceKey::from_space(lhs),
                rhs: FastSpaceKey::from_space(rhs),
                axes: axes_key.clone(),
                scope: scope.fast(),
                access,
            };
            if let Some(resolution) = self.fast.get(&fast_key) {
                self.stats.hits += 1;
                self.stats.fast_hits += 1;
                let resolution = resolution.clone();
                self.remember_last(
                    &rule_key,
                    dst,
                    lhs,
                    rhs,
                    axes,
                    scope,
                    access,
                    None,
                    &resolution,
                );
                return Ok(resolution);
            }

            let full_key = FullKey {
                rule: rule_key.clone(),
                dst: FullSpaceKey::from_space(dst)?,
                lhs: FullSpaceKey::from_space(lhs)?,
                rhs: FullSpaceKey::from_space(rhs)?,
                axes: axes_key.clone(),
                scope: scope.full()?,
                access,
            };
            if let Some(resolution) = self.resolved.get(&full_key) {
                self.stats.hits += 1;
                let resolution = resolution.clone();
                self.touch(&full_key);
                self.fast_insert(fast_key, resolution.clone());
                self.remember_last(
                    &rule_key,
                    dst,
                    lhs,
                    rhs,
                    axes,
                    scope,
                    access,
                    Some(full_key),
                    &resolution,
                );
                return Ok(resolution);
            }

            self.stats.misses += 1;
            let global = global_contraction_resolutions::<RuleKey>();
            if let Some(resolution) = global
                .read()
                .expect("global contraction resolution cache poisoned")
                .get(&full_key)
                .cloned()
            {
                self.resolved.insert(full_key.clone(), resolution.clone());
                if let Some(max_entries) = self.policy.max_entries() {
                    self.lru_order.push_back(full_key.clone());
                    self.enforce_lru_limit(max_entries);
                }
                self.fast_insert(fast_key, resolution.clone());
                self.remember_last(
                    &rule_key,
                    dst,
                    lhs,
                    rhs,
                    axes,
                    scope,
                    access,
                    Some(full_key),
                    &resolution,
                );
                return Ok(resolution);
            }
            let resolution = resolve_cold()?;
            global
                .write()
                .expect("global contraction resolution cache poisoned")
                .entry(full_key.clone())
                .or_insert_with(|| resolution.clone());
            self.resolved.insert(full_key.clone(), resolution.clone());
            if let Some(max_entries) = self.policy.max_entries() {
                self.lru_order.push_back(full_key.clone());
                self.enforce_lru_limit(max_entries);
            }
            self.fast_insert(fast_key, resolution.clone());
            self.remember_last(
                &rule_key,
                dst,
                lhs,
                rhs,
                axes,
                scope,
                access,
                Some(full_key),
                &resolution,
            );
            return Ok(resolution);
        }

        self.stats.misses += 1;
        resolve_cold()
    }

    #[allow(clippy::too_many_arguments)]
    fn remember_last(
        &mut self,
        rule_key: &RuleKey,
        dst: &DynamicFusionMapSpace,
        lhs: &DynamicFusionMapSpace,
        rhs: &DynamicFusionMapSpace,
        axes: TensorContractSpec<'_>,
        scope: LiveResolutionScope<'_>,
        access: OperandAccess,
        full_key: Option<FullKey<RuleKey>>,
        resolution: &Resolution,
    ) {
        self.last.insert(
            0,
            LastEntry {
                rule: rule_key.clone(),
                dst: LastSpace::from_space(dst),
                lhs: LastSpace::from_space(lhs),
                rhs: LastSpace::from_space(rhs),
                axes: RawAxes::from_axes(axes),
                scope: scope.pinned(),
                access,
                full_key,
                resolution: resolution.clone(),
            },
        );
        self.last.truncate(LAST_RING_CAPACITY);
    }
}

/// Uncached resolution: the route test and the plan compile are one step.
fn resolve<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
    compile_structure: impl FnOnce()
        -> Result<Option<Arc<TensorContractStructure<f64>>>, OperationError>,
    compile_dynamic: impl FnOnce() -> Result<Arc<FusionContractPlan>, OperationError>,
) -> Result<Resolution, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    if !axes.lhs_conjugate() && !axes.rhs_conjugate() {
        if is_core_form_fusion_block_contract(rule, dst, lhs, rhs, axes)?
            && !rhs_contract_requires_twist(rule, rhs, axes)?
        {
            let plan = compile_fusion_block_contract_plan_validated(rule, dst, lhs, rhs, axes)?;
            if plan.is_fully_direct() {
                return Ok(Resolution::Core(Arc::new(plan)));
            }
        }
        return Ok(Resolution::DynamicTree(compile_dynamic()?));
    }
    if let Some(structure) = compile_structure()? {
        return Ok(Resolution::Structure(structure));
    }
    Ok(Resolution::DynamicTree(compile_dynamic()?))
}

/// True when the fermionic supertrace twist can be nontrivial: such
/// contractions take the dynamic route, where the twist is applied during
/// rhs materialization; the core direct-GEMM route stays
/// coefficient-free (TensorKit mul! parity).
pub(crate) fn rhs_contract_requires_twist<R>(
    rule: &R,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
) -> Result<bool, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    rhs_contract_homspace_requires_twist(rule, rhs.homspace(), axes)
}

pub(crate) fn rhs_contract_homspace_requires_twist<R>(
    rule: &R,
    rhs: &FusionTreeHomSpace,
    axes: TensorContractSpec<'_>,
) -> Result<bool, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    if rule.braiding_style() != tenet_core::BraidingStyleKind::Fermionic {
        return Ok(false);
    }
    for &axis in axes.rhs_contracting_axes() {
        if external_axis_is_dual(rhs, axis)? {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::fusion::prepare_tensorcontract_fusion_plan_dyn;
    use crate::BoundDynamicFusionMapSpace;
    use tenet_core::{
        FermionParityFusionRule, FusionProductSpace, FusionTreeHomSpace, ProductFusionRuleExt,
        SU2FusionRule, SU2Irrep, SectorId, SectorLeg, U1FusionRule, U1Irrep,
    };
    use tenet_operations::axis::OutputAxisOrder;

    // Two-leg-per-side U(1) matrix space (three charges) in a chosen bond
    // dimension `deg`. Each call builds a fresh hom-space `Arc` (see
    // `from_degeneracy_shapes`), so content-equal spaces from separate calls
    // carry distinct pointers — the case that made the old raw-pointer key miss.
    fn u1_matrix_space(rule: &U1FusionRule, deg: usize) -> DynamicFusionMapSpace {
        let sectors = [
            U1Irrep::new(-1).sector_id(),
            U1Irrep::new(0).sector_id(),
            U1Irrep::new(1).sector_id(),
        ];
        let leg = || SectorLeg::new(sectors.map(|sector| (sector, deg)), false);
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg()]),
            FusionProductSpace::new([leg(), leg()]),
        );
        let count = hom.fusion_tree_keys(rule).len();
        DynamicFusionMapSpace::from_degeneracy_shapes(rule, hom, vec![vec![deg; 4]; count]).unwrap()
    }

    fn single_sector_matrix_space<R>(
        rule: &R,
        sector: SectorId,
        codomain_dual: bool,
        domain_dual: bool,
    ) -> DynamicFusionMapSpace
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(sector, 1)], codomain_dual)]),
            FusionProductSpace::new([SectorLeg::new([(sector, 1)], domain_dual)]),
        );
        let count = hom.fusion_tree_keys(rule).len();
        DynamicFusionMapSpace::from_degeneracy_shapes(rule, hom, vec![vec![1, 1]; count]).unwrap()
    }

    #[test]
    fn rhs_twist_requirement_uses_external_domain_dual_and_product_parity() {
        let fermion = FermionParityFusionRule;
        let odd = SectorId::new(1);
        let cases = [
            (false, false, 0, false),
            (true, false, 0, true),
            (false, false, 1, true),
            (false, true, 1, false),
        ];
        for (codomain_dual, domain_dual, rhs_axis, expected) in cases {
            let rhs = single_sector_matrix_space(&fermion, odd, codomain_dual, domain_dual);
            let rhs_axes = [rhs_axis];
            let axes = TensorContractSpec::with_default_output_order(&[0], &rhs_axes);
            // What: codomain uses its stored dual flag, while domain external
            // duality is the inverse of its stored flag.
            assert_eq!(
                rhs_contract_requires_twist(&fermion, &rhs, axes).unwrap(),
                expected
            );
        }

        let fp_u1 = FermionParityFusionRule.product(U1FusionRule);
        let odd_charge = fp_u1.encode_sector(odd, U1Irrep::new(0).sector_id());
        let product = fp_u1.product(SU2FusionRule);
        let odd_product =
            product.encode_sector(odd_charge, SU2Irrep::from_twice_spin(0).sector_id());
        let rhs = single_sector_matrix_space(&product, odd_product, true, false);
        // What: a bosonic U(1) x SU(2) component does not erase the odd fZ2
        // twist on an externally dual product-sector axis.
        assert!(rhs_contract_requires_twist(
            &product,
            &rhs,
            TensorContractSpec::with_default_output_order(&[0], &[0]),
        )
        .unwrap());
    }

    // New capability of content re-keying: content-equal spaces backed by
    // distinct Arcs now collapse to one fast key (a would-be `FullKey` hit),
    // where the old `Arc::as_ptr` key saw two distinct keys and missed.
    #[test]
    fn fast_space_key_hits_on_equal_content_across_distinct_arcs() {
        // What: `FastSpaceKey`/`FullSpaceKey` embed the block structure's
        // interned `content_id`, so this relies on the tenet-core intern
        // table handing `a` and `b`'s content-equal structures the *same*
        // id. A concurrent `reset_global_operation_caches` (which chains
        // into `reset_core_intern_tables`) landing between the two builds
        // would evict the first entry and re-intern the second with a
        // fresh id, breaking the `assert_eq!` below.
        let _guard = crate::test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let rule = Arc::new(U1FusionRule);
        let a = u1_matrix_space(&rule, 3);
        let b = u1_matrix_space(&rule, 3);
        // Precondition: independently built content-equal spaces do carry
        // distinct hom-space Arcs, so this is a genuine ABA-shaped case.
        assert!(!Arc::ptr_eq(a.homspace_arc(), b.homspace_arc()));
        // Content keys agree, and they agree exactly where `FullKey` does —
        // `FastKey` and `FullKey` share one content-equivalence class.
        assert_eq!(FastSpaceKey::from_space(&a), FastSpaceKey::from_space(&b));
        assert_eq!(
            FullSpaceKey::from_space(&a).unwrap(),
            FullSpaceKey::from_space(&b).unwrap()
        );
        // Distinct content (different bond dimension) still keys apart.
        let c = u1_matrix_space(&rule, 4);
        assert_ne!(FastSpaceKey::from_space(&a), FastSpaceKey::from_space(&c));
    }

    // Pins the invariant the fast key's safety argument rests on: a
    // `DynamicTree` plan is a pure function of (rank, axes, conj) and carries no
    // degeneracy, so the same swap contraction at χ=1 and χ=3 compiles to a
    // byte-identical plan. If a future plan change made plans degeneracy-
    // dependent, this loud failure flags that the Why-not comment on
    // [`FastKey`] no longer holds.
    #[test]
    fn dynamic_tree_plan_is_content_independent_across_chi() {
        let rule = Arc::new(U1FusionRule);
        // Swap contraction (permutes rhs), forcing the tree-transform route.
        let axes =
            TensorContractSpec::new(&[3, 2], &[0, 1], OutputAxisOrder::from_axes(&[0, 1, 2, 3]));
        let plan = |deg| {
            let raw = u1_matrix_space(rule.as_ref(), deg);
            let space =
                BoundDynamicFusionMapSpace::bind_multiplicity_free(raw, Arc::clone(&rule)).unwrap();
            prepare_tensorcontract_fusion_plan_dyn(&space, &space, &space, axes).unwrap()
        };
        assert_eq!(plan(1), plan(3));
    }
}
