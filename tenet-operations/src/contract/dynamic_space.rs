use std::collections::HashMap;
use std::sync::Arc;

use tenet_core::{
    BlockKey, BlockStructure, CoreError, FusionTensorMapSpace, FusionTreeBlockKey,
    FusionTreeHomSpace, MultiplicityFreeRigidSymbols,
};

use crate::cache::BlockStructureCacheKey;
use crate::tree_transform::build_tree_pair_transform_group_plan;
use crate::{OperationError, TreeTransformOperationKey};

use super::fusion::{contracted_fusion_tree_basis_matches, TensorContractFusionExplicitPlan};
use super::structure::TensorContractAxisPlan;

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

    pub(super) fn cache_key(&self) -> Result<DynamicFusionMapSpaceCacheKey, OperationError> {
        DynamicFusionMapSpaceCacheKey::from_dynamic_space(self)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(super) struct DynamicFusionMapSpaceCacheKey {
    nout: usize,
    nin: usize,
    homspace: FusionTreeHomSpace,
    structure: BlockStructureCacheKey,
}

impl DynamicFusionMapSpaceCacheKey {
    pub(super) fn from_typed_space<const NOUT: usize, const NIN: usize>(
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
