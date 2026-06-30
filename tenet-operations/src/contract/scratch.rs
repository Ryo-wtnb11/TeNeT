use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;

use num_traits::Zero;
use tenet_core::{
    BlockKey, BlockStructure, CoreError, FusionTensorMapSpace, FusionTreeBlockKey,
    FusionTreeHomSpace, MultiplicityFreeRigidSymbols,
};

use crate::axis::{OwnedTensorContractAxisSpec, TensorContractAxisSpec};
use crate::cache::BlockStructureCacheKey;
use crate::tree_transform::build_tree_pair_transform_group_plan;
use crate::{OperationError, TreeTransformOperationKey, TreeTransformRuleCacheKey};

use super::fusion::{contracted_fusion_tree_basis_matches, TensorContractFusionExplicitPlan};
use super::structure::{TensorContractAxisPlan, TensorContractBlockSpec};

/// Internal dynamic-rank fusion space used for TensorKit-style temporary
/// materialization. Public tensors remain const-generic; source/output tree
/// transforms that change the codomain/domain split are absorbed here.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DynamicFusionMapSpace {
    nout: usize,
    nin: usize,
    homspace: FusionTreeHomSpace,
    subblock_structure: Arc<BlockStructure>,
}

impl DynamicFusionMapSpace {
    pub(crate) fn from_typed<const NOUT: usize, const NIN: usize>(
        space: &FusionTensorMapSpace<NOUT, NIN>,
    ) -> Self {
        Self {
            nout: NOUT,
            nin: NIN,
            homspace: space.homspace().clone(),
            subblock_structure: Arc::clone(space.subblock_structure()),
        }
    }

    pub(crate) fn transformed_from_typed<R, const NOUT: usize, const NIN: usize>(
        rule: &R,
        source: &FusionTensorMapSpace<NOUT, NIN>,
        operation: &TreeTransformOperationKey,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        let (codomain_axes, domain_axes) = tree_transform_operation_axes(operation);
        let nout = codomain_axes.len();
        let nin = domain_axes.len();
        let homspace = source
            .homspace()
            .permute(rule, codomain_axes, domain_axes)
            .map_err(OperationError::from_core_preserving_context)?;
        let plan = build_tree_pair_transform_group_plan(
            rule,
            operation.clone(),
            source.subblock_structure(),
        )?;
        let mut blocks = Vec::<(BlockKey, Vec<usize>)>::new();
        for spec in plan.specs() {
            let src_count = spec.src_keys().len();
            for (dst_row, dst_key) in spec.dst_keys().iter().enumerate() {
                let mut dst_shape = None::<Vec<usize>>;
                for (src_column, src_key) in spec.src_keys().iter().enumerate() {
                    let coefficient =
                        spec.coefficients_src_by_dst()[src_column + dst_row * src_count];
                    if coefficient == 0.0 {
                        continue;
                    }
                    let src_block = source
                        .subblock_structure()
                        .block_by_key(src_key)
                        .map_err(OperationError::from_core_preserving_context)?;
                    let candidate = selected_shape(src_block.shape(), codomain_axes, domain_axes)?;
                    if let Some(existing) = &dst_shape {
                        if existing != &candidate {
                            return Err(OperationError::ShapeMismatch {
                                dst: existing.clone(),
                                src: candidate,
                            });
                        }
                    } else {
                        dst_shape = Some(candidate);
                    }
                }
                let dst_shape = dst_shape.ok_or(OperationError::EmptyTransformBlock)?;
                blocks.push((dst_key.clone(), dst_shape));
            }
        }
        let subblock_structure = Arc::new(BlockStructure::packed_column_major_with_keys(
            nout + nin,
            blocks,
        )?);
        Ok(Self {
            nout,
            nin,
            homspace,
            subblock_structure,
        })
    }

    pub(crate) fn canonical_dst<R>(
        rule: &R,
        lhs: &Self,
        rhs: &Self,
        plan: &TensorContractFusionExplicitPlan,
        output_dst: Option<&Self>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        let nout = plan.canonical_dst_nout();
        let nin = plan.canonical_dst_nin();
        let axes = plan.canonical_axes().as_spec();
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

        let mut inferred_shapes = infer_canonical_dst_shapes(rule, lhs, rhs, &axis_plan)?;
        if let Some(output_dst) = output_dst {
            infer_canonical_dst_shapes_from_output(
                rule,
                &homspace,
                plan,
                output_dst,
                &mut inferred_shapes,
            )?;
        }
        let mut blocks = Vec::<(BlockKey, Vec<usize>)>::new();
        for key in homspace.fusion_tree_keys(rule) {
            let shape = inferred_shapes.get(&key).cloned().ok_or_else(|| {
                OperationError::MissingBlockKey {
                    key: BlockKey::from(key.clone()),
                }
            })?;
            blocks.push((BlockKey::from(key), shape));
        }
        let subblock_structure = Arc::new(BlockStructure::packed_column_major_with_keys(
            nout + nin,
            blocks,
        )?);
        Ok(Self {
            nout,
            nin,
            homspace,
            subblock_structure,
        })
    }

    #[inline]
    pub(crate) fn nout(&self) -> usize {
        self.nout
    }

    #[inline]
    pub(crate) fn rank(&self) -> usize {
        self.nout + self.nin
    }

    #[inline]
    pub(crate) fn homspace(&self) -> &FusionTreeHomSpace {
        &self.homspace
    }

    #[inline]
    pub(crate) fn structure(&self) -> &Arc<BlockStructure> {
        &self.subblock_structure
    }

    pub(crate) fn required_len(&self) -> Result<usize, CoreError> {
        self.subblock_structure.required_len()
    }

    pub(crate) fn find_subblock_index(&self, key: &FusionTreeBlockKey) -> Option<usize> {
        self.subblock_structure
            .find_block_index_by_fusion_tree_key(key)
    }

    fn cache_key(&self) -> Result<DynamicFusionMapSpaceCacheKey, OperationError> {
        DynamicFusionMapSpaceCacheKey::from_dynamic_space(self)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct DynamicFusionMapSpaceCacheKey {
    nout: usize,
    nin: usize,
    homspace: FusionTreeHomSpace,
    structure: BlockStructureCacheKey,
}

impl DynamicFusionMapSpaceCacheKey {
    fn from_typed_space<const NOUT: usize, const NIN: usize>(
        space: &FusionTensorMapSpace<NOUT, NIN>,
    ) -> Result<Self, OperationError> {
        Ok(Self {
            nout: NOUT,
            nin: NIN,
            homspace: space.homspace().clone(),
            structure: BlockStructureCacheKey::from_structure(space.subblock_structure())?,
        })
    }

    fn from_dynamic_space(space: &DynamicFusionMapSpace) -> Result<Self, OperationError> {
        Ok(Self {
            nout: space.nout,
            nin: space.nin,
            homspace: space.homspace.clone(),
            structure: BlockStructureCacheKey::from_structure(space.subblock_structure.as_ref())?,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct DynamicFusionTransformedSpaceCacheKey<RuleKey> {
    rule: RuleKey,
    source: DynamicFusionMapSpaceCacheKey,
    operation: TreeTransformOperationKey,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct DynamicFusionCanonicalDstSpaceCacheKey<RuleKey> {
    rule: RuleKey,
    lhs: DynamicFusionMapSpaceCacheKey,
    rhs: DynamicFusionMapSpaceCacheKey,
    axes: OwnedTensorContractAxisSpec,
    canonical_dst_nout: usize,
    canonical_dst_nin: usize,
    output_transform: TreeTransformOperationKey,
    output_dst: Option<DynamicFusionMapSpaceCacheKey>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TensorContractFusionSpaceCacheStats {
    transformed_hits: usize,
    transformed_misses: usize,
    canonical_dst_hits: usize,
    canonical_dst_misses: usize,
}

impl TensorContractFusionSpaceCacheStats {
    #[inline]
    pub fn transformed_hits(self) -> usize {
        self.transformed_hits
    }

    #[inline]
    pub fn transformed_misses(self) -> usize {
        self.transformed_misses
    }

    #[inline]
    pub fn canonical_dst_hits(self) -> usize {
        self.canonical_dst_hits
    }

    #[inline]
    pub fn canonical_dst_misses(self) -> usize {
        self.canonical_dst_misses
    }
}

#[derive(Clone, Debug)]
pub(crate) struct DynamicFusionSpaceCache<RuleKey> {
    transformed: HashMap<DynamicFusionTransformedSpaceCacheKey<RuleKey>, DynamicFusionMapSpace>,
    canonical_dst: HashMap<DynamicFusionCanonicalDstSpaceCacheKey<RuleKey>, DynamicFusionMapSpace>,
    stats: TensorContractFusionSpaceCacheStats,
}

impl<RuleKey> Default for DynamicFusionSpaceCache<RuleKey> {
    fn default() -> Self {
        Self {
            transformed: HashMap::new(),
            canonical_dst: HashMap::new(),
            stats: TensorContractFusionSpaceCacheStats::default(),
        }
    }
}

impl<RuleKey> DynamicFusionSpaceCache<RuleKey>
where
    RuleKey: Clone + Eq + Hash,
{
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.transformed.len() + self.canonical_dst.len()
    }

    #[inline]
    pub(crate) fn transformed_len(&self) -> usize {
        self.transformed.len()
    }

    #[inline]
    pub(crate) fn canonical_dst_len(&self) -> usize {
        self.canonical_dst.len()
    }

    #[inline]
    pub(crate) fn stats(&self) -> TensorContractFusionSpaceCacheStats {
        self.stats
    }

    pub(crate) fn transformed_from_typed<R, const NOUT: usize, const NIN: usize>(
        &mut self,
        rule: &R,
        source: &FusionTensorMapSpace<NOUT, NIN>,
        operation: &TreeTransformOperationKey,
    ) -> Result<DynamicFusionMapSpace, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        let key = DynamicFusionTransformedSpaceCacheKey {
            rule: rule.tree_transform_rule_cache_key(),
            source: DynamicFusionMapSpaceCacheKey::from_typed_space(source)?,
            operation: operation.clone(),
        };
        if let Some(space) = self.transformed.get(&key) {
            self.stats.transformed_hits += 1;
            return Ok(space.clone());
        }
        self.stats.transformed_misses += 1;
        let space = DynamicFusionMapSpace::transformed_from_typed(rule, source, operation)?;
        self.transformed.insert(key, space.clone());
        Ok(space)
    }

    pub(crate) fn canonical_dst<R>(
        &mut self,
        rule: &R,
        lhs: &DynamicFusionMapSpace,
        rhs: &DynamicFusionMapSpace,
        plan: &TensorContractFusionExplicitPlan,
        output_dst: Option<&DynamicFusionMapSpace>,
    ) -> Result<DynamicFusionMapSpace, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        let key = DynamicFusionCanonicalDstSpaceCacheKey {
            rule: rule.tree_transform_rule_cache_key(),
            lhs: lhs.cache_key()?,
            rhs: rhs.cache_key()?,
            axes: plan.canonical_axes().clone(),
            canonical_dst_nout: plan.canonical_dst_nout(),
            canonical_dst_nin: plan.canonical_dst_nin(),
            output_transform: plan.output_transform().clone(),
            output_dst: output_dst
                .map(DynamicFusionMapSpace::cache_key)
                .transpose()?,
        };
        if let Some(space) = self.canonical_dst.get(&key) {
            self.stats.canonical_dst_hits += 1;
            return Ok(space.clone());
        }
        self.stats.canonical_dst_misses += 1;
        let space = DynamicFusionMapSpace::canonical_dst(rule, lhs, rhs, plan, output_dst)?;
        self.canonical_dst.insert(key, space.clone());
        Ok(space)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct DynamicFusionScratch<T> {
    space: DynamicFusionMapSpace,
    data: Vec<T>,
}

impl<T> DynamicFusionScratch<T>
where
    T: Clone + Zero,
{
    pub(crate) fn zeroed(space: DynamicFusionMapSpace) -> Result<Self, OperationError> {
        let len = space.required_len()?;
        Ok(Self {
            space,
            data: vec![T::zero(); len],
        })
    }

    pub(crate) fn fill_zero(&mut self) {
        self.data.fill(T::zero());
    }
}

impl<T> DynamicFusionScratch<T> {
    #[inline]
    pub(crate) fn space(&self) -> &DynamicFusionMapSpace {
        &self.space
    }

    #[inline]
    pub(crate) fn data(&self) -> &[T] {
        &self.data
    }

    #[inline]
    pub(crate) fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}

#[derive(Clone, Debug)]
pub(crate) struct DynamicFusionScratchWorkspace<T> {
    lhs: Option<DynamicFusionScratch<T>>,
    rhs: Option<DynamicFusionScratch<T>>,
    dst: Option<DynamicFusionScratch<T>>,
}

impl<T> Default for DynamicFusionScratchWorkspace<T> {
    fn default() -> Self {
        Self {
            lhs: None,
            rhs: None,
            dst: None,
        }
    }
}

impl<T> DynamicFusionScratchWorkspace<T>
where
    T: Clone + Zero,
{
    pub(crate) fn prepare_lhs(
        &mut self,
        space: DynamicFusionMapSpace,
    ) -> Result<&mut DynamicFusionScratch<T>, OperationError> {
        prepare_scratch_slot(&mut self.lhs, space)
    }

    pub(crate) fn prepare_rhs(
        &mut self,
        space: DynamicFusionMapSpace,
    ) -> Result<&mut DynamicFusionScratch<T>, OperationError> {
        prepare_scratch_slot(&mut self.rhs, space)
    }

    pub(crate) fn prepare_dst(
        &mut self,
        space: DynamicFusionMapSpace,
    ) -> Result<&mut DynamicFusionScratch<T>, OperationError> {
        prepare_scratch_slot(&mut self.dst, space)
    }

    pub(crate) fn lhs(&self) -> &DynamicFusionScratch<T> {
        self.lhs
            .as_ref()
            .expect("lhs dynamic scratch prepared before replay")
    }

    pub(crate) fn rhs(&self) -> &DynamicFusionScratch<T> {
        self.rhs
            .as_ref()
            .expect("rhs dynamic scratch prepared before replay")
    }

    pub(crate) fn dst(&self) -> &DynamicFusionScratch<T> {
        self.dst
            .as_ref()
            .expect("dst dynamic scratch prepared before replay")
    }

    pub(crate) fn lhs_rhs(&self) -> (&DynamicFusionScratch<T>, &DynamicFusionScratch<T>) {
        (self.lhs(), self.rhs())
    }

    pub(crate) fn lhs_rhs_dst_mut(
        &mut self,
    ) -> (
        &DynamicFusionScratch<T>,
        &DynamicFusionScratch<T>,
        &mut DynamicFusionScratch<T>,
    ) {
        let Self { lhs, rhs, dst } = self;
        (
            lhs.as_ref()
                .expect("lhs dynamic scratch prepared before replay"),
            rhs.as_ref()
                .expect("rhs dynamic scratch prepared before replay"),
            dst.as_mut()
                .expect("dst dynamic scratch prepared before replay"),
        )
    }
}

fn prepare_scratch_slot<T>(
    slot: &mut Option<DynamicFusionScratch<T>>,
    space: DynamicFusionMapSpace,
) -> Result<&mut DynamicFusionScratch<T>, OperationError>
where
    T: Clone + Zero,
{
    match slot {
        Some(scratch) if scratch.space == space => {
            scratch.fill_zero();
        }
        _ => {
            *slot = Some(DynamicFusionScratch::zeroed(space)?);
        }
    }
    Ok(slot
        .as_mut()
        .expect("dynamic scratch slot prepared before return"))
}

pub(crate) fn tensorcontract_dynamic_canonical_fusion_block_specs<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractAxisSpec<'_>,
) -> Result<Vec<TensorContractBlockSpec>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let axis_plan = TensorContractAxisPlan::compile(lhs.rank(), rhs.rank(), dst.rank(), axes)?;
    let expected_homspace = FusionTreeHomSpace::tensorcontract_homspace(
        rule,
        lhs.homspace(),
        rhs.homspace(),
        axes.lhs_contracting_axes(),
        axes.rhs_contracting_axes(),
        axis_plan.output_axes.as_slice(),
        dst.nout(),
    )
    .map_err(OperationError::from_core_preserving_context)?;
    if expected_homspace != *dst.homspace() {
        return Err(OperationError::StructureMismatch { tensor: "dst" });
    }
    if !is_canonical_dynamic_source_contract(lhs, rhs, &axis_plan) {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "dynamic fusion contraction expects canonical source tree-pair transforms",
        });
    }

    let mut specs = Vec::new();
    for lhs_index in 0..lhs.structure().block_count() {
        let lhs_block = lhs.structure().block(lhs_index)?;
        let BlockKey::FusionTree(lhs_key) = lhs_block.key() else {
            continue;
        };
        let lhs_external = lhs_key.external_sectors(rule);
        for rhs_index in 0..rhs.structure().block_count() {
            let rhs_block = rhs.structure().block(rhs_index)?;
            let BlockKey::FusionTree(rhs_key) = rhs_block.key() else {
                continue;
            };
            let rhs_external = rhs_key.external_sectors(rule);
            if !contracted_external_sectors_match(
                &lhs_external,
                &rhs_external,
                axis_plan.lhs_contracting_axes.as_slice(),
                axis_plan.rhs_contracting_axes.as_slice(),
            ) {
                continue;
            }
            if !contracted_fusion_tree_basis_matches(
                rule,
                lhs_key.domain_tree(),
                rhs_key.codomain_tree(),
            ) {
                continue;
            }
            let dst_key = FusionTreeBlockKey::pair(
                lhs_key.codomain_tree().clone(),
                rhs_key.domain_tree().clone(),
            );
            let dst_index = dst.find_subblock_index(&dst_key).ok_or_else(|| {
                OperationError::MissingBlockKey {
                    key: BlockKey::from(dst_key.clone()),
                }
            })?;
            specs.push(TensorContractBlockSpec::with_coefficient(
                dst_index,
                lhs_index,
                rhs_index,
                rule.scalar_one(),
            ));
        }
    }
    Ok(specs)
}

fn infer_canonical_dst_shapes<R>(
    rule: &R,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axis_plan: &TensorContractAxisPlan,
) -> Result<HashMap<FusionTreeBlockKey, Vec<usize>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let mut shapes = HashMap::<FusionTreeBlockKey, Vec<usize>>::new();
    for lhs_index in 0..lhs.structure().block_count() {
        let lhs_block = lhs.structure().block(lhs_index)?;
        let BlockKey::FusionTree(lhs_key) = lhs_block.key() else {
            continue;
        };
        let lhs_external = lhs_key.external_sectors(rule);
        for rhs_index in 0..rhs.structure().block_count() {
            let rhs_block = rhs.structure().block(rhs_index)?;
            let BlockKey::FusionTree(rhs_key) = rhs_block.key() else {
                continue;
            };
            let rhs_external = rhs_key.external_sectors(rule);
            if !contracted_external_sectors_match(
                &lhs_external,
                &rhs_external,
                axis_plan.lhs_contracting_axes.as_slice(),
                axis_plan.rhs_contracting_axes.as_slice(),
            ) {
                continue;
            }
            if !contracted_fusion_tree_basis_matches(
                rule,
                lhs_key.domain_tree(),
                rhs_key.codomain_tree(),
            ) {
                continue;
            }
            let dst_key = FusionTreeBlockKey::pair(
                lhs_key.codomain_tree().clone(),
                rhs_key.domain_tree().clone(),
            );
            let shape = axis_plan
                .lhs_open_axes
                .iter()
                .map(|&axis| lhs_block.shape()[axis])
                .chain(
                    axis_plan
                        .rhs_open_axes
                        .iter()
                        .map(|&axis| rhs_block.shape()[axis]),
                )
                .collect::<Vec<_>>();
            if let Some(existing) = shapes.get(&dst_key) {
                if existing != &shape {
                    return Err(OperationError::ShapeMismatch {
                        dst: existing.clone(),
                        src: shape,
                    });
                }
            } else {
                shapes.insert(dst_key, shape);
            }
        }
    }
    Ok(shapes)
}

fn infer_canonical_dst_shapes_from_output<R>(
    rule: &R,
    canonical_homspace: &FusionTreeHomSpace,
    plan: &TensorContractFusionExplicitPlan,
    output_dst: &DynamicFusionMapSpace,
    shapes: &mut HashMap<FusionTreeBlockKey, Vec<usize>>,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let canonical_rank = plan.canonical_dst_nout() + plan.canonical_dst_nin();
    let dummy_blocks = canonical_homspace
        .fusion_tree_keys(rule)
        .into_iter()
        .map(|key| (BlockKey::from(key), vec![1; canonical_rank]));
    let dummy_structure =
        BlockStructure::packed_column_major_with_keys(canonical_rank, dummy_blocks)?;
    let transform_plan = build_tree_pair_transform_group_plan(
        rule,
        plan.output_transform().clone(),
        &dummy_structure,
    )?;
    let (codomain_axes, domain_axes) = tree_transform_operation_axes(plan.output_transform());
    let output_axes = codomain_axes
        .iter()
        .chain(domain_axes)
        .copied()
        .collect::<Vec<_>>();
    for spec in transform_plan.specs() {
        let src_count = spec.src_keys().len();
        for (src_column, src_key) in spec.src_keys().iter().enumerate() {
            let BlockKey::FusionTree(src_tree_key) = src_key else {
                continue;
            };
            for (dst_row, dst_key) in spec.dst_keys().iter().enumerate() {
                let coefficient = spec.coefficients_src_by_dst()[src_column + dst_row * src_count];
                if coefficient == 0.0 {
                    continue;
                }
                let Ok(dst_block) = output_dst.structure().block_by_key(dst_key) else {
                    continue;
                };
                let candidate = invert_selected_shape(
                    dst_block.shape(),
                    &output_axes,
                    canonical_rank,
                    "output",
                )?;
                merge_inferred_shape(shapes, src_tree_key.clone(), candidate)?;
            }
        }
    }
    Ok(())
}

fn merge_inferred_shape(
    shapes: &mut HashMap<FusionTreeBlockKey, Vec<usize>>,
    key: FusionTreeBlockKey,
    candidate: Vec<usize>,
) -> Result<(), OperationError> {
    if let Some(existing) = shapes.get(&key) {
        if existing != &candidate {
            return Err(OperationError::ShapeMismatch {
                dst: existing.clone(),
                src: candidate,
            });
        }
    } else {
        shapes.insert(key, candidate);
    }
    Ok(())
}

fn is_canonical_dynamic_source_contract(
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axis_plan: &TensorContractAxisPlan,
) -> bool {
    let lhs_domain_axes = (lhs.nout()..lhs.rank()).collect::<Vec<_>>();
    let rhs_codomain_axes = (0..rhs.nout()).collect::<Vec<_>>();
    axis_plan.lhs_contracting_axes == lhs_domain_axes
        && axis_plan.rhs_contracting_axes == rhs_codomain_axes
}

fn contracted_external_sectors_match(
    lhs_external: &[tenet_core::SectorId],
    rhs_external: &[tenet_core::SectorId],
    lhs_axes: &[usize],
    rhs_axes: &[usize],
) -> bool {
    lhs_axes
        .iter()
        .zip(rhs_axes)
        .all(|(&lhs_axis, &rhs_axis)| lhs_external[lhs_axis] == rhs_external[rhs_axis])
}

fn tree_transform_operation_axes(operation: &TreeTransformOperationKey) -> (&[usize], &[usize]) {
    match operation {
        TreeTransformOperationKey::Transpose {
            codomain_permutation,
            domain_permutation,
        }
        | TreeTransformOperationKey::Permute {
            codomain_permutation,
            domain_permutation,
        }
        | TreeTransformOperationKey::Braid {
            codomain_permutation,
            domain_permutation,
            ..
        } => (
            codomain_permutation.as_slice(),
            domain_permutation.as_slice(),
        ),
    }
}

fn selected_shape(
    shape: &[usize],
    codomain_axes: &[usize],
    domain_axes: &[usize],
) -> Result<Vec<usize>, OperationError> {
    let mut selected = Vec::with_capacity(codomain_axes.len() + domain_axes.len());
    for &axis in codomain_axes.iter().chain(domain_axes) {
        let dim = shape.get(axis).copied().ok_or_else(|| {
            let mut axes = Vec::with_capacity(codomain_axes.len() + domain_axes.len());
            axes.extend_from_slice(codomain_axes);
            axes.extend_from_slice(domain_axes);
            OperationError::InvalidAxisSet {
                tensor: "src",
                axes,
                rank: shape.len(),
            }
        })?;
        selected.push(dim);
    }
    Ok(selected)
}

fn invert_selected_shape(
    selected_shape: &[usize],
    axes: &[usize],
    rank: usize,
    tensor: &'static str,
) -> Result<Vec<usize>, OperationError> {
    if selected_shape.len() != axes.len() {
        return Err(OperationError::RankMismatch {
            expected: axes.len(),
            actual: selected_shape.len(),
        });
    }
    let mut seen = vec![false; rank];
    let mut shape = vec![0; rank];
    for (&axis, &dim) in axes.iter().zip(selected_shape) {
        if axis >= rank || seen[axis] {
            return Err(OperationError::InvalidAxisSet {
                tensor,
                axes: axes.to_vec(),
                rank,
            });
        }
        seen[axis] = true;
        shape[axis] = dim;
    }
    if seen.iter().any(|&value| !value) {
        return Err(OperationError::InvalidAxisSet {
            tensor,
            axes: axes.to_vec(),
            rank,
        });
    }
    Ok(shape)
}
