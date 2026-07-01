use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;

use num_traits::{One, Zero};
use tenet_core::{
    BlockKey, BlockStructure, FusionTreeHomSpace, FusionTreeKey, MultiplicityFreeRigidSymbols,
    SectorId,
};

use crate::axis::{OwnedTensorContractAxisSpec, TensorContractAxisSpec};
use crate::cache::BlockStructureCacheKey;
use crate::strided::{
    column_major_strides_isize, column_major_strides_usize, element_count, error as strided_error,
    offset_to_isize, strides_to_isize,
};
use crate::structure_identity::validate_structure_identity;
use crate::{
    DenseBlockScalar, OperationError, RecouplingCoefficientAction, TreeTransformRuleCacheKey,
};

use super::backend::TensorContractBackend;
use super::dynamic_space::DynamicFusionMapSpace;
use super::fusion::{reject_fusion_contract_conjugation, rhs_contract_twist_factor};
use super::profile::TensorContractFusionProfile;
use super::structure::{TensorContractAxisPlan, TensorContractStructure};

#[allow(clippy::too_many_arguments)]
pub(crate) fn tensorcontract_canonical_fusion_blocks_into_raw<B, R, D>(
    backend: &mut B,
    workspace: &mut B::Workspace,
    rule: &R,
    dst_space: &DynamicFusionMapSpace,
    dst_data: &mut [D],
    lhs_space: &DynamicFusionMapSpace,
    lhs_data: &[D],
    rhs_space: &DynamicFusionMapSpace,
    rhs_data: &[D],
    axes: TensorContractAxisSpec<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    B: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
{
    let plan =
        CanonicalFusionBlockContractPlan::compile(rule, dst_space, lhs_space, rhs_space, axes)?;
    let mut fusion_workspace = CanonicalFusionBlockContractWorkspace::<D>::default();
    plan.execute_raw(
        backend,
        workspace,
        &mut fusion_workspace,
        dst_space.structure(),
        dst_data,
        lhs_space.structure(),
        lhs_data,
        rhs_space.structure(),
        rhs_data,
        alpha,
        beta,
    )
}

pub(crate) fn is_canonical_fusion_block_contract<R>(
    rule: &R,
    dst_space: &DynamicFusionMapSpace,
    lhs_space: &DynamicFusionMapSpace,
    rhs_space: &DynamicFusionMapSpace,
    axes: TensorContractAxisSpec<'_>,
) -> Result<bool, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    reject_fusion_contract_conjugation(axes)?;
    let axis_plan = TensorContractAxisPlan::compile(
        lhs_space.rank(),
        rhs_space.rank(),
        dst_space.rank(),
        axes,
    )?;
    if !is_canonical_source(lhs_space, rhs_space, &axis_plan)
        || !is_canonical_output(dst_space, lhs_space, rhs_space, &axis_plan)
    {
        return Ok(false);
    }
    let expected_homspace = FusionTreeHomSpace::tensorcontract_homspace(
        rule,
        lhs_space.homspace(),
        rhs_space.homspace(),
        axes.lhs_contracting_axes(),
        axes.rhs_contracting_axes(),
        axis_plan.output_axes.as_slice(),
        dst_space.nout(),
    )
    .map_err(OperationError::from_core_preserving_context)?;
    if expected_homspace != *dst_space.homspace() {
        return Err(OperationError::StructureMismatch { tensor: "dst" });
    }
    Ok(true)
}

fn validate_canonical_compose<R>(
    rule: &R,
    dst_space: &DynamicFusionMapSpace,
    lhs_space: &DynamicFusionMapSpace,
    rhs_space: &DynamicFusionMapSpace,
    axes: TensorContractAxisSpec<'_>,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    if is_canonical_fusion_block_contract(rule, dst_space, lhs_space, rhs_space, axes)? {
        Ok(())
    } else {
        Err(OperationError::UnsupportedTensorContractScope {
            message: "canonical fusion-block contraction requires canonical source and output axes",
        })
    }
}

fn is_canonical_source(
    lhs_space: &DynamicFusionMapSpace,
    rhs_space: &DynamicFusionMapSpace,
    axis_plan: &TensorContractAxisPlan,
) -> bool {
    let lhs_domain_axes = (lhs_space.nout()..lhs_space.rank()).collect::<Vec<_>>();
    let rhs_codomain_axes = (0..rhs_space.nout()).collect::<Vec<_>>();
    axis_plan.lhs_contracting_axes == lhs_domain_axes
        && axis_plan.rhs_contracting_axes == rhs_codomain_axes
}

fn is_canonical_output(
    dst_space: &DynamicFusionMapSpace,
    lhs_space: &DynamicFusionMapSpace,
    rhs_space: &DynamicFusionMapSpace,
    axis_plan: &TensorContractAxisPlan,
) -> bool {
    let output_rank = lhs_space.nout() + (rhs_space.rank() - rhs_space.nout());
    let canonical_output_axes = (0..output_rank).collect::<Vec<_>>();
    dst_space.nout() == lhs_space.nout() && axis_plan.output_axes == canonical_output_axes
}

#[derive(Clone, Debug)]
pub(crate) struct CanonicalFusionBlockContractWorkspace<T> {
    buffers: FusionBlockContractBuffers<T>,
}

impl<T> Default for CanonicalFusionBlockContractWorkspace<T> {
    fn default() -> Self {
        Self {
            buffers: FusionBlockContractBuffers::default(),
        }
    }
}

#[derive(Clone, Debug)]
struct FusionBlockContractBuffers<T> {
    lhs: Vec<T>,
    rhs: Vec<T>,
    dst: Vec<T>,
}

impl<T> Default for FusionBlockContractBuffers<T> {
    fn default() -> Self {
        Self {
            lhs: Vec::new(),
            rhs: Vec::new(),
            dst: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CanonicalFusionBlockContractPlan {
    dst_structure: Arc<BlockStructure>,
    lhs_structure: Arc<BlockStructure>,
    rhs_structure: Arc<BlockStructure>,
    groups: Vec<CanonicalFusionBlockContractGroupPlan>,
}

impl CanonicalFusionBlockContractPlan {
    pub(crate) fn compile<R>(
        rule: &R,
        dst_space: &DynamicFusionMapSpace,
        lhs_space: &DynamicFusionMapSpace,
        rhs_space: &DynamicFusionMapSpace,
        axes: TensorContractAxisSpec<'_>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        reject_fusion_contract_conjugation(axes)?;
        validate_canonical_compose(rule, dst_space, lhs_space, rhs_space, axes)?;
        let axis_plan = TensorContractAxisPlan::compile(
            lhs_space.rank(),
            rhs_space.rank(),
            dst_space.rank(),
            axes,
        )?;

        let lhs_layout = FusionBlockMatrixLayout::compile(rule, lhs_space, None)?;
        let rhs_layout = FusionBlockMatrixLayout::compile(
            rule,
            rhs_space,
            Some(&axis_plan.rhs_contracting_axes),
        )?;
        let dst_layout = FusionBlockMatrixLayout::compile(rule, dst_space, None)?;

        let mut groups = Vec::new();
        for lhs_group in lhs_layout.groups {
            let Some(rhs_group) = rhs_layout.group(lhs_group.coupled) else {
                continue;
            };
            let Some(dst_group) = dst_layout.group(lhs_group.coupled) else {
                continue;
            };
            groups.push(CanonicalFusionBlockContractGroupPlan::compile(
                lhs_group,
                rhs_group.clone(),
                dst_group.clone(),
            )?);
        }
        Ok(Self {
            dst_structure: Arc::clone(dst_space.structure()),
            lhs_structure: Arc::clone(lhs_space.structure()),
            rhs_structure: Arc::clone(rhs_space.structure()),
            groups,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_raw<B, D>(
        &self,
        backend: &mut B,
        workspace: &mut B::Workspace,
        fusion_workspace: &mut CanonicalFusionBlockContractWorkspace<D>,
        dst_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        lhs_structure: &Arc<BlockStructure>,
        lhs_data: &[D],
        rhs_structure: &Arc<BlockStructure>,
        rhs_data: &[D],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        B: TensorContractBackend<D, f64>,
        D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    {
        self.validate_replay_structures(dst_structure, lhs_structure, rhs_structure)?;
        scale_all_blocks(dst_structure, dst_data, beta)?;

        for group in &self.groups {
            fusion_workspace
                .buffers
                .prepare(group.lhs.rows, group.lhs.cols, group.rhs.cols)?;
            pack_group(
                &group.lhs,
                lhs_structure,
                lhs_data,
                &mut fusion_workspace.buffers.lhs,
            )?;
            pack_group(
                &group.rhs,
                rhs_structure,
                rhs_data,
                &mut fusion_workspace.buffers.rhs,
            )?;
            matmul_group_plan(
                backend,
                workspace,
                group,
                &fusion_workspace.buffers.lhs,
                &fusion_workspace.buffers.rhs,
                &mut fusion_workspace.buffers.dst,
            )?;
            scatter_group(
                &group.dst,
                dst_structure,
                dst_data,
                &fusion_workspace.buffers.dst,
                alpha,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_raw_profiled<B, D>(
        &self,
        backend: &mut B,
        workspace: &mut B::Workspace,
        fusion_workspace: &mut CanonicalFusionBlockContractWorkspace<D>,
        dst_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        lhs_structure: &Arc<BlockStructure>,
        lhs_data: &[D],
        rhs_structure: &Arc<BlockStructure>,
        rhs_data: &[D],
        alpha: D,
        beta: D,
        profile: &mut TensorContractFusionProfile,
    ) -> Result<(), OperationError>
    where
        B: TensorContractBackend<D, f64>,
        D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    {
        let total_start = std::time::Instant::now();

        let start = std::time::Instant::now();
        self.validate_replay_structures(dst_structure, lhs_structure, rhs_structure)?;
        profile.canonical_validate += start.elapsed();

        let start = std::time::Instant::now();
        scale_all_blocks(dst_structure, dst_data, beta)?;
        profile.canonical_scale += start.elapsed();

        for group in &self.groups {
            profile.canonical_contract_groups += 1;

            let start = std::time::Instant::now();
            fusion_workspace
                .buffers
                .prepare(group.lhs.rows, group.lhs.cols, group.rhs.cols)?;
            profile.canonical_workspace_prepare += start.elapsed();

            let start = std::time::Instant::now();
            pack_group(
                &group.lhs,
                lhs_structure,
                lhs_data,
                &mut fusion_workspace.buffers.lhs,
            )?;
            profile.canonical_pack_lhs += start.elapsed();

            let start = std::time::Instant::now();
            pack_group(
                &group.rhs,
                rhs_structure,
                rhs_data,
                &mut fusion_workspace.buffers.rhs,
            )?;
            profile.canonical_pack_rhs += start.elapsed();

            let start = std::time::Instant::now();
            matmul_group_plan(
                backend,
                workspace,
                group,
                &fusion_workspace.buffers.lhs,
                &fusion_workspace.buffers.rhs,
                &mut fusion_workspace.buffers.dst,
            )?;
            profile.canonical_matmul += start.elapsed();

            let start = std::time::Instant::now();
            scatter_group(
                &group.dst,
                dst_structure,
                dst_data,
                &fusion_workspace.buffers.dst,
                alpha,
            )?;
            profile.canonical_scatter += start.elapsed();
        }

        profile.canonical_contract_total += total_start.elapsed();
        Ok(())
    }

    fn validate_replay_structures(
        &self,
        dst_structure: &Arc<BlockStructure>,
        lhs_structure: &Arc<BlockStructure>,
        rhs_structure: &Arc<BlockStructure>,
    ) -> Result<(), OperationError> {
        validate_structure_identity("dst", &self.dst_structure, dst_structure)?;
        validate_structure_identity("lhs", &self.lhs_structure, lhs_structure)?;
        validate_structure_identity("rhs", &self.rhs_structure, rhs_structure)
    }
}

#[derive(Clone, Debug)]
struct CanonicalFusionBlockContractGroupPlan {
    lhs: FusionBlockMatrixGroup,
    rhs: FusionBlockMatrixGroup,
    dst: FusionBlockMatrixGroup,
    matmul_dst_structure: Arc<BlockStructure>,
    matmul_lhs_structure: Arc<BlockStructure>,
    matmul_rhs_structure: Arc<BlockStructure>,
    matmul_structure: TensorContractStructure,
}

impl CanonicalFusionBlockContractGroupPlan {
    fn compile(
        lhs: FusionBlockMatrixGroup,
        rhs: FusionBlockMatrixGroup,
        dst: FusionBlockMatrixGroup,
    ) -> Result<Self, OperationError> {
        if lhs.cols != rhs.rows {
            return Err(OperationError::ShapeMismatch {
                dst: vec![lhs.cols],
                src: vec![rhs.rows],
            });
        }
        if dst.rows != lhs.rows || dst.cols != rhs.cols {
            return Err(OperationError::ShapeMismatch {
                dst: vec![dst.rows, dst.cols],
                src: vec![lhs.rows, rhs.cols],
            });
        }

        let matmul_lhs_structure = Arc::new(BlockStructure::trivial(&[lhs.rows, lhs.cols])?);
        let matmul_rhs_structure = Arc::new(BlockStructure::trivial(&[lhs.cols, rhs.cols])?);
        let matmul_dst_structure = Arc::new(BlockStructure::trivial(&[lhs.rows, rhs.cols])?);
        let matmul_structure = TensorContractStructure::compile_structures(
            &matmul_dst_structure,
            &matmul_lhs_structure,
            &matmul_rhs_structure,
            TensorContractAxisSpec::canonical(&[1], &[0]),
        )?;
        Ok(Self {
            lhs,
            rhs,
            dst,
            matmul_dst_structure,
            matmul_lhs_structure,
            matmul_rhs_structure,
            matmul_structure,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct CanonicalFusionBlockContractCacheStats {
    hits: usize,
    misses: usize,
}

impl CanonicalFusionBlockContractCacheStats {
    #[inline]
    pub(crate) fn hits(self) -> usize {
        self.hits
    }

    #[inline]
    pub(crate) fn misses(self) -> usize {
        self.misses
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CanonicalFusionBlockContractCache<RuleKey> {
    plans: HashMap<
        CanonicalFusionBlockContractCacheKey<RuleKey>,
        Arc<CanonicalFusionBlockContractPlan>,
    >,
    stats: CanonicalFusionBlockContractCacheStats,
}

impl<RuleKey> Default for CanonicalFusionBlockContractCache<RuleKey> {
    fn default() -> Self {
        Self {
            plans: HashMap::new(),
            stats: CanonicalFusionBlockContractCacheStats::default(),
        }
    }
}

impl<RuleKey> CanonicalFusionBlockContractCache<RuleKey>
where
    RuleKey: Clone + Eq + Hash,
{
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.plans.len()
    }

    #[inline]
    pub(crate) fn stats(&self) -> CanonicalFusionBlockContractCacheStats {
        self.stats
    }

    pub(crate) fn get_or_compile<R>(
        &mut self,
        rule: &R,
        dst_space: &DynamicFusionMapSpace,
        lhs_space: &DynamicFusionMapSpace,
        rhs_space: &DynamicFusionMapSpace,
        axes: TensorContractAxisSpec<'_>,
    ) -> Result<Arc<CanonicalFusionBlockContractPlan>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        let key = CanonicalFusionBlockContractCacheKey::from_spaces(
            rule, dst_space, lhs_space, rhs_space, axes,
        )?;
        if self.plans.get(&key).is_some() {
            self.stats.hits += 1;
        } else {
            self.stats.misses += 1;
            let plan = CanonicalFusionBlockContractPlan::compile(
                rule, dst_space, lhs_space, rhs_space, axes,
            )?;
            self.plans.insert(key.clone(), Arc::new(plan));
        }
        Ok(Arc::clone(self.plans.get(&key).expect(
            "canonical fusion block contract plan inserted before replay",
        )))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct CanonicalFusionBlockContractCacheKey<RuleKey> {
    rule: RuleKey,
    dst_nout: usize,
    dst_homspace: FusionTreeHomSpace,
    dst_structure: BlockStructureCacheKey,
    lhs_nout: usize,
    lhs_homspace: FusionTreeHomSpace,
    lhs_structure: BlockStructureCacheKey,
    rhs_nout: usize,
    rhs_homspace: FusionTreeHomSpace,
    rhs_structure: BlockStructureCacheKey,
    axes: OwnedTensorContractAxisSpec,
}

impl<RuleKey> CanonicalFusionBlockContractCacheKey<RuleKey>
where
    RuleKey: Clone + Eq + Hash,
{
    fn from_spaces<R>(
        rule: &R,
        dst: &DynamicFusionMapSpace,
        lhs: &DynamicFusionMapSpace,
        rhs: &DynamicFusionMapSpace,
        axes: TensorContractAxisSpec<'_>,
    ) -> Result<Self, OperationError>
    where
        R: TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        let axis_plan = TensorContractAxisPlan::compile(lhs.rank(), rhs.rank(), dst.rank(), axes)?;
        Ok(Self {
            rule: rule.tree_transform_rule_cache_key(),
            dst_nout: dst.nout(),
            dst_homspace: dst.homspace().clone(),
            dst_structure: BlockStructureCacheKey::from_structure(dst.structure())?,
            lhs_nout: lhs.nout(),
            lhs_homspace: lhs.homspace().clone(),
            lhs_structure: BlockStructureCacheKey::from_structure(lhs.structure())?,
            rhs_nout: rhs.nout(),
            rhs_homspace: rhs.homspace().clone(),
            rhs_structure: BlockStructureCacheKey::from_structure(rhs.structure())?,
            axes: OwnedTensorContractAxisSpec::new_with_conjugation(
                axis_plan.lhs_contracting_axes,
                axis_plan.rhs_contracting_axes,
                axis_plan.output_axes,
                axis_plan.lhs_conjugate,
                axis_plan.rhs_conjugate,
            ),
        })
    }
}

impl<T> FusionBlockContractBuffers<T>
where
    T: Clone + Zero,
{
    fn prepare(
        &mut self,
        lhs_rows: usize,
        contracted: usize,
        rhs_cols: usize,
    ) -> Result<(), OperationError> {
        let lhs_len = lhs_rows
            .checked_mul(contracted)
            .ok_or(OperationError::ElementCountOverflow)?;
        let rhs_len = contracted
            .checked_mul(rhs_cols)
            .ok_or(OperationError::ElementCountOverflow)?;
        let dst_len = lhs_rows
            .checked_mul(rhs_cols)
            .ok_or(OperationError::ElementCountOverflow)?;
        self.lhs.resize(lhs_len, T::zero());
        self.lhs.fill(T::zero());
        self.rhs.resize(rhs_len, T::zero());
        self.rhs.fill(T::zero());
        self.dst.resize(dst_len, T::zero());
        self.dst.fill(T::zero());
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct FusionBlockMatrixLayout {
    groups: Vec<FusionBlockMatrixGroup>,
}

impl FusionBlockMatrixLayout {
    fn compile<R>(
        rule: &R,
        space: &DynamicFusionMapSpace,
        rhs_contracting_axes: Option<&[usize]>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        let mut builders = Vec::<FusionBlockMatrixGroupBuilder>::new();
        let mut group_indices = HashMap::<SectorId, usize>::new();
        for block_index in 0..space.structure().block_count() {
            let block = space.structure().block(block_index)?;
            let BlockKey::FusionTree(key) = block.key() else {
                return Err(OperationError::ExpectedFusionTreeBlock {
                    tensor: "fusion",
                    index: block_index,
                });
            };
            let coupled = coupled_sector(rule, key.codomain_tree());
            if coupled != coupled_sector(rule, key.domain_tree()) {
                return Err(OperationError::FusionTreeGroupMismatch {
                    tensor: "fusion",
                    index: block_index,
                });
            }
            let group_index = if let Some(&group_index) = group_indices.get(&coupled) {
                group_index
            } else {
                let group_index = builders.len();
                group_indices.insert(coupled, group_index);
                builders.push(FusionBlockMatrixGroupBuilder::new(coupled));
                group_index
            };
            let row_dim = element_count(&block.shape()[..space.nout()])?;
            let col_dim = element_count(&block.shape()[space.nout()..])?;
            builders[group_index].add_tree_pair(
                key.codomain_tree().clone(),
                row_dim,
                key.domain_tree().clone(),
                col_dim,
                block_index,
            )?;
        }
        let mut groups = Vec::with_capacity(builders.len());
        for builder in builders {
            groups.push(builder.finish(rule, space, rhs_contracting_axes)?);
        }
        Ok(Self { groups })
    }

    fn group(&self, coupled: SectorId) -> Option<&FusionBlockMatrixGroup> {
        self.groups.iter().find(|group| group.coupled == coupled)
    }
}

#[derive(Clone, Debug)]
struct FusionBlockMatrixGroupBuilder {
    coupled: SectorId,
    row_offsets: HashMap<FusionTreeKey, TreeMatrixOffset>,
    col_offsets: HashMap<FusionTreeKey, TreeMatrixOffset>,
    blocks: Vec<usize>,
    rows: usize,
    cols: usize,
}

impl FusionBlockMatrixGroupBuilder {
    fn new(coupled: SectorId) -> Self {
        Self {
            coupled,
            row_offsets: HashMap::new(),
            col_offsets: HashMap::new(),
            blocks: Vec::new(),
            rows: 0,
            cols: 0,
        }
    }

    fn add_tree_pair(
        &mut self,
        row_tree: FusionTreeKey,
        row_dim: usize,
        col_tree: FusionTreeKey,
        col_dim: usize,
        block_index: usize,
    ) -> Result<(), OperationError> {
        match self.row_offsets.get(&row_tree) {
            Some(offset) if offset.dim != row_dim => {
                return Err(OperationError::ShapeMismatch {
                    dst: vec![offset.dim],
                    src: vec![row_dim],
                });
            }
            Some(_) => {}
            None => {
                let offset = self.rows;
                self.rows = self
                    .rows
                    .checked_add(row_dim)
                    .ok_or(OperationError::ElementCountOverflow)?;
                self.row_offsets.insert(
                    row_tree,
                    TreeMatrixOffset {
                        offset,
                        dim: row_dim,
                    },
                );
            }
        }
        match self.col_offsets.get(&col_tree) {
            Some(offset) if offset.dim != col_dim => {
                return Err(OperationError::ShapeMismatch {
                    dst: vec![offset.dim],
                    src: vec![col_dim],
                });
            }
            Some(_) => {}
            None => {
                let offset = self.cols;
                self.cols = self
                    .cols
                    .checked_add(col_dim)
                    .ok_or(OperationError::ElementCountOverflow)?;
                self.col_offsets.insert(
                    col_tree,
                    TreeMatrixOffset {
                        offset,
                        dim: col_dim,
                    },
                );
            }
        }
        self.blocks.push(block_index);
        Ok(())
    }

    fn finish<R>(
        self,
        rule: &R,
        space: &DynamicFusionMapSpace,
        rhs_contracting_axes: Option<&[usize]>,
    ) -> Result<FusionBlockMatrixGroup, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        let mut subblocks = Vec::with_capacity(self.blocks.len());
        for block_index in self.blocks {
            let block = space.structure().block(block_index)?;
            let BlockKey::FusionTree(key) = block.key() else {
                return Err(OperationError::ExpectedFusionTreeBlock {
                    tensor: "fusion",
                    index: block_index,
                });
            };
            let row = self
                .row_offsets
                .get(key.codomain_tree())
                .expect("row tree offset collected before finish");
            let col = self
                .col_offsets
                .get(key.domain_tree())
                .expect("column tree offset collected before finish");
            let mut matrix_strides = Vec::<isize>::with_capacity(block.shape().len());
            matrix_strides.extend(column_major_strides_isize(&block.shape()[..space.nout()])?);
            let domain_strides = column_major_strides_usize(&block.shape()[space.nout()..])?;
            for stride in domain_strides {
                let matrix_stride = stride
                    .checked_mul(self.rows)
                    .ok_or(OperationError::ElementCountOverflow)?;
                matrix_strides.push(isize::try_from(matrix_stride).map_err(|_| {
                    OperationError::StrideOverflow {
                        value: matrix_stride,
                    }
                })?);
            }
            let matrix_offset = col
                .offset
                .checked_mul(self.rows)
                .and_then(|offset| offset.checked_add(row.offset))
                .ok_or(OperationError::ElementCountOverflow)?;
            let coefficient = if let Some(rhs_contracting_axes) = rhs_contracting_axes {
                rhs_contract_twist_factor(
                    rule,
                    space.homspace(),
                    rhs_contracting_axes,
                    key.codomain_tree(),
                )?
            } else {
                rule.scalar_one()
            };
            subblocks.push(FusionSubblockMatrixLayout {
                block_index,
                matrix_offset,
                matrix_strides,
                coefficient,
            });
        }
        Ok(FusionBlockMatrixGroup {
            coupled: self.coupled,
            rows: self.rows,
            cols: self.cols,
            subblocks,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct TreeMatrixOffset {
    offset: usize,
    dim: usize,
}

#[derive(Clone, Debug)]
struct FusionBlockMatrixGroup {
    coupled: SectorId,
    rows: usize,
    cols: usize,
    subblocks: Vec<FusionSubblockMatrixLayout>,
}

#[derive(Clone, Debug)]
struct FusionSubblockMatrixLayout {
    block_index: usize,
    matrix_offset: usize,
    matrix_strides: Vec<isize>,
    coefficient: f64,
}

fn coupled_sector<R>(rule: &R, tree: &FusionTreeKey) -> SectorId
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    tree.coupled().unwrap_or_else(|| rule.vacuum())
}

fn pack_group<T>(
    group: &FusionBlockMatrixGroup,
    structure: &BlockStructure,
    data: &[T],
    packed: &mut [T],
) -> Result<(), OperationError>
where
    T: Copy
        + std::ops::Mul<T, Output = T>
        + RecouplingCoefficientAction<f64>
        + strided_kernel::MaybeSendSync,
{
    for layout in &group.subblocks {
        let block = structure.block(layout.block_index)?;
        let src_strides = strides_to_isize(block.strides())?;
        let mut dst = strided_kernel::StridedViewMut::new(
            packed,
            block.shape(),
            &layout.matrix_strides,
            offset_to_isize(layout.matrix_offset)?,
        )
        .map_err(strided_error)?;
        let src = strided_kernel::StridedView::<T>::new(
            data,
            block.shape(),
            &src_strides,
            offset_to_isize(block.offset())?,
        )
        .map_err(strided_error)?;
        if layout.coefficient == 1.0 {
            strided_kernel::copy_into(&mut dst, &src).map_err(strided_error)?;
        } else {
            strided_kernel::copy_scale(&mut dst, &src, T::coefficient_as_data(layout.coefficient))
                .map_err(strided_error)?;
        }
    }
    Ok(())
}

fn scatter_group<T>(
    group: &FusionBlockMatrixGroup,
    structure: &BlockStructure,
    data: &mut [T],
    packed: &[T],
    alpha: T,
) -> Result<(), OperationError>
where
    T: Copy
        + std::ops::Add<T, Output = T>
        + std::ops::Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + strided_kernel::MaybeSendSync,
{
    for layout in &group.subblocks {
        let block = structure.block(layout.block_index)?;
        let dst_strides = strides_to_isize(block.strides())?;
        let mut dst = strided_kernel::StridedViewMut::new(
            data,
            block.shape(),
            &dst_strides,
            offset_to_isize(block.offset())?,
        )
        .map_err(strided_error)?;
        let src = strided_kernel::StridedView::<T>::new(
            packed,
            block.shape(),
            &layout.matrix_strides,
            offset_to_isize(layout.matrix_offset)?,
        )
        .map_err(strided_error)?;
        strided_kernel::axpy(&mut dst, &src, alpha).map_err(strided_error)?;
    }
    Ok(())
}

fn scale_all_blocks<T>(
    structure: &BlockStructure,
    data: &mut [T],
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy + std::ops::Mul<T, Output = T> + strided_kernel::MaybeSendSync,
{
    for block_index in 0..structure.block_count() {
        let block = structure.block(block_index)?;
        let strides = strides_to_isize(block.strides())?;
        let mut dst = strided_kernel::StridedViewMut::new(
            data,
            block.shape(),
            &strides,
            offset_to_isize(block.offset())?,
        )
        .map_err(strided_error)?;
        scale_view(&mut dst, beta)?;
    }
    Ok(())
}

fn scale_view<T>(
    dst: &mut strided_kernel::StridedViewMut<'_, T>,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy + std::ops::Mul<T, Output = T> + strided_kernel::MaybeSendSync,
{
    let scalar = [beta];
    let zero_strides = vec![0isize; dst.ndim()];
    let beta_view = strided_kernel::StridedView::<T>::new(&scalar, dst.dims(), &zero_strides, 0)
        .map_err(strided_error)?;
    strided_kernel::mul(dst, &beta_view).map_err(strided_error)
}

fn matmul_group_plan<B, D>(
    backend: &mut B,
    workspace: &mut B::Workspace,
    group: &CanonicalFusionBlockContractGroupPlan,
    lhs: &[D],
    rhs: &[D],
    dst: &mut [D],
) -> Result<(), OperationError>
where
    B: TensorContractBackend<D, f64>,
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
{
    backend.tensorcontract_structure_into_raw(
        workspace,
        &group.matmul_structure,
        &group.matmul_dst_structure,
        &group.matmul_lhs_structure,
        &group.matmul_rhs_structure,
        dst,
        lhs,
        rhs,
        D::one(),
        D::zero(),
    )
}
