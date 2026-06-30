use std::hash::Hash;

use tenet_core::TensorMap;

use crate::axis::{OwnedTensorContractAxisSpec, TensorContractAxisSpec};
use crate::backend::DenseTreeTransformOperations;
use crate::cache::{TensorContractStructureCache, TensorContractStructureCacheKey};
use crate::{DenseBlockScalar, OperationError, RecouplingCoefficientAction};

use super::backend::TensorContractBackend;
use super::structure::{TensorContractAxisPlan, TensorContractStructure};

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct TensorContractPlanKey {
    axes: OwnedTensorContractAxisSpec,
}

impl TensorContractPlanKey {
    pub fn from_axes(
        lhs_rank: usize,
        rhs_rank: usize,
        dst_rank: usize,
        axes: TensorContractAxisSpec<'_>,
    ) -> Result<Self, OperationError> {
        let axis_plan = TensorContractAxisPlan::compile(lhs_rank, rhs_rank, dst_rank, axes)?;
        Ok(Self {
            axes: OwnedTensorContractAxisSpec::new(
                axis_plan.lhs_contracting_axes,
                axis_plan.rhs_contracting_axes,
                axis_plan.output_axes,
            ),
        })
    }

    #[inline]
    pub fn axes(&self) -> &OwnedTensorContractAxisSpec {
        &self.axes
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TensorContractCacheStats {
    structure_hits: usize,
    structure_misses: usize,
}

impl TensorContractCacheStats {
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
pub struct TensorContractCache<PlanKey = TensorContractPlanKey> {
    structures: TensorContractStructureCache<f64, PlanKey>,
    stats: TensorContractCacheStats,
}

impl<PlanKey> Default for TensorContractCache<PlanKey> {
    fn default() -> Self {
        Self {
            structures: TensorContractStructureCache::default(),
            stats: TensorContractCacheStats::default(),
        }
    }
}

impl<PlanKey> TensorContractCache<PlanKey>
where
    PlanKey: Clone + Eq + Hash,
{
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn structure_len(&self) -> usize {
        self.structures.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.structures.is_empty()
    }

    #[inline]
    pub fn stats(&self) -> TensorContractCacheStats {
        self.stats
    }

    pub fn reset_stats(&mut self) {
        self.stats = TensorContractCacheStats::default();
    }
}

impl TensorContractCache<TensorContractPlanKey> {
    pub fn get_or_compile<
        TDst,
        TLhs,
        TRhs,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const LHS_NOUT: usize,
        const LHS_NIN: usize,
        const RHS_NOUT: usize,
        const RHS_NIN: usize,
        SDst,
        SLhs,
        SRhs,
    >(
        &mut self,
        dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst>,
        lhs: &TensorMap<TLhs, LHS_NOUT, LHS_NIN, SLhs>,
        rhs: &TensorMap<TRhs, RHS_NOUT, RHS_NIN, SRhs>,
        axes: TensorContractAxisSpec<'_>,
    ) -> Result<&TensorContractStructure, OperationError> {
        let plan_key = TensorContractPlanKey::from_axes(
            lhs.structure().rank(),
            rhs.structure().rank(),
            dst.structure().rank(),
            axes,
        )?;
        let structure_key = TensorContractStructureCacheKey::from_structures(
            plan_key.clone(),
            dst.structure(),
            lhs.structure(),
            rhs.structure(),
        )?;
        if self.structures.get(&structure_key).is_some() {
            self.stats.structure_hits += 1;
        } else {
            self.stats.structure_misses += 1;
            let structure =
                TensorContractStructure::compile(dst, lhs, rhs, plan_key.axes().as_spec())?;
            self.structures.insert(structure_key.clone(), structure);
        }
        Ok(self
            .structures
            .get(&structure_key)
            .expect("tensor contract structure inserted before replay"))
    }
}

#[derive(Debug)]
pub struct TensorContractExecutionContext<D, B = DenseTreeTransformOperations>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    B: TensorContractBackend<D, f64>,
{
    backend: B,
    workspace: B::Workspace,
    cache: TensorContractCache,
}

impl<D, B> TensorContractExecutionContext<D, B>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    B: TensorContractBackend<D, f64>,
{
    pub fn with_parts(backend: B, workspace: B::Workspace, cache: TensorContractCache) -> Self {
        Self {
            backend,
            workspace,
            cache,
        }
    }

    #[inline]
    pub fn backend(&self) -> &B {
        &self.backend
    }

    #[inline]
    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    #[inline]
    pub fn workspace(&self) -> &B::Workspace {
        &self.workspace
    }

    #[inline]
    pub fn workspace_mut(&mut self) -> &mut B::Workspace {
        &mut self.workspace
    }

    #[inline]
    pub fn cache(&self) -> &TensorContractCache {
        &self.cache
    }

    #[inline]
    pub fn cache_mut(&mut self) -> &mut TensorContractCache {
        &mut self.cache
    }

    pub fn into_parts(self) -> (B, B::Workspace, TensorContractCache) {
        (self.backend, self.workspace, self.cache)
    }
}

impl<D, B> TensorContractExecutionContext<D, B>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    B: TensorContractBackend<D, f64>,
    B::Workspace: Default,
{
    pub fn new(backend: B) -> Self {
        Self::with_parts(backend, B::Workspace::default(), TensorContractCache::new())
    }
}

impl<D, B> Default for TensorContractExecutionContext<D, B>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    B: TensorContractBackend<D, f64> + Default,
    B::Workspace: Default,
{
    fn default() -> Self {
        Self::new(B::default())
    }
}

impl<D, B> TensorContractExecutionContext<D, B>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    B: TensorContractBackend<D, f64>,
{
    pub fn tensorcontract_into<
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const LHS_NOUT: usize,
        const LHS_NIN: usize,
        const RHS_NOUT: usize,
        const RHS_NIN: usize,
        SDst,
        SLhs,
        SRhs,
    >(
        &mut self,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
        lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs>,
        rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs>,
        axes: TensorContractAxisSpec<'_>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError> {
        let Self {
            backend,
            workspace,
            cache,
        } = self;
        let structure = cache.get_or_compile(dst, lhs, rhs, axes)?;
        backend.tensorcontract_structure_into(workspace, structure, dst, lhs, rhs, alpha, beta)
    }
}

pub fn tensorcontract_into_with_context<
    B,
    D,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const LHS_NOUT: usize,
    const LHS_NIN: usize,
    const RHS_NOUT: usize,
    const RHS_NIN: usize,
    SDst,
    SLhs,
    SRhs,
>(
    context: &mut TensorContractExecutionContext<D, B>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs>,
    axes: TensorContractAxisSpec<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    B: TensorContractBackend<D, f64>,
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
{
    context.tensorcontract_into(dst, lhs, rhs, axes, alpha, beta)
}
