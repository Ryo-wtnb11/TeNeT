use std::collections::HashMap;
use std::sync::Arc;

use tenet_core::{
    BlockKey, BlockStructure, CoreError, FusionTensorMapSpace, FusionTreeBlockKey,
    FusionTreeHomSpace, MultiplicityFreeRigidSymbols, SectorId,
};

use crate::tree_transform::build_tree_pair_transform_group_plan;
use crate::{OperationError, TreeTransformOperationKey};

/// Builds scratch structures in the coupled-sector matrix layout. Scratch
/// spaces enumerate the full tree set of their hom spaces, so the coupled
/// grid is always complete; there is no other layout.
fn scratch_subblock_structure<R>(
    rule: &R,
    nout: usize,
    rank: usize,
    blocks: Vec<(BlockKey, Vec<usize>)>,
) -> Result<BlockStructure, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let mut tree_blocks = Vec::with_capacity(blocks.len());
    for (index, (key, shape)) in blocks.iter().enumerate() {
        match key {
            BlockKey::FusionTree(tree) => tree_blocks.push((tree.clone(), shape.clone())),
            BlockKey::Dense => {
                return Err(OperationError::ExpectedFusionTreeBlock {
                    tensor: "scratch",
                    index,
                })
            }
        }
    }
    BlockStructure::coupled_sector_matrix_with_keys(rule, nout, rank, tree_blocks)
        .map_err(OperationError::from_core_preserving_context)
}

use super::fusion::{contracted_fusion_tree_basis_matches, TensorContractFusionExplicitPlan};
use super::structure::TensorContractAxisPlan;

/// Internal dynamic-rank fusion space used for TensorKit-style temporary
/// materialization. Public tensors remain const-generic; source/output tree
/// transforms that change the codomain/domain split are absorbed here.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DynamicFusionMapSpace {
    nout: usize,
    nin: usize,
    homspace: Arc<FusionTreeHomSpace>,
    subblock_structure: Arc<BlockStructure>,
}

impl DynamicFusionMapSpace {
    pub(crate) fn from_typed<const NOUT: usize, const NIN: usize>(
        space: &FusionTensorMapSpace<NOUT, NIN>,
    ) -> Self {
        Self {
            nout: NOUT,
            nin: NIN,
            homspace: Arc::clone(space.homspace_arc()),
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
        // Enumerate the FULL tree set of the permuted hom space (TensorKit
        // semantics): trees the transform coefficients never reach stay as
        // structural zeros in the scratch buffer, keeping every coupled
        // sector grid complete so the coupled matrix layout always applies.
        let src_dims = axis_sector_dims(rule, source.subblock_structure())?;
        let src_axes = codomain_axes
            .iter()
            .chain(domain_axes.iter())
            .copied()
            .collect::<Vec<_>>();
        let keys = homspace.fusion_tree_keys(rule);
        let mut blocks = Vec::<(BlockKey, Vec<usize>)>::with_capacity(keys.len());
        for key in keys {
            let sectors = key.external_sectors(rule);
            let mut shape = Vec::with_capacity(src_axes.len());
            for (out_axis, &src_axis) in src_axes.iter().enumerate() {
                let dim = src_dims[src_axis].get(&sectors[out_axis]).copied().ok_or(
                    OperationError::StructureMismatch {
                        tensor: "transformed scratch",
                    },
                )?;
                shape.push(dim);
            }
            blocks.push((BlockKey::from(key), shape));
        }
        let subblock_structure =
            Arc::new(scratch_subblock_structure(rule, nout, nout + nin, blocks)?);
        Ok(Self {
            nout,
            nin,
            homspace: Arc::new(homspace),
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
        // Complete the tree set: keys the contraction pairing never produces
        // still get a subblock (structural zero) so the coupled grid is full.
        let lhs_dims = axis_sector_dims(rule, lhs.structure())?;
        let rhs_dims = axis_sector_dims(rule, rhs.structure())?;
        let lhs_open = axis_plan.lhs_open_axes.clone();
        let rhs_open = axis_plan.rhs_open_axes.clone();
        let keys = homspace.fusion_tree_keys(rule);
        let mut blocks = Vec::<(BlockKey, Vec<usize>)>::with_capacity(keys.len());
        for key in keys {
            let shape = match inferred_shapes.get(&key) {
                Some(shape) => shape.clone(),
                None => {
                    let sectors = key.external_sectors(rule);
                    let mut shape = Vec::with_capacity(lhs_open.len() + rhs_open.len());
                    for (out_axis, &sector) in sectors.iter().enumerate() {
                        let dim = if out_axis < lhs_open.len() {
                            lhs_dims[lhs_open[out_axis]].get(&sector).copied()
                        } else {
                            rhs_dims[rhs_open[out_axis - lhs_open.len()]]
                                .get(&sector)
                                .copied()
                        };
                        shape.push(dim.ok_or(OperationError::StructureMismatch {
                            tensor: "canonical contraction scratch",
                        })?);
                    }
                    shape
                }
            };
            blocks.push((BlockKey::from(key), shape));
        }
        let subblock_structure =
            Arc::new(scratch_subblock_structure(rule, nout, nout + nin, blocks)?);

        Ok(Self {
            nout,
            nin,
            homspace: Arc::new(homspace),
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

    /// Shared hom-space handle for pointer-identity fast paths in replay
    /// caches.
    pub(crate) fn homspace_arc(&self) -> &Arc<FusionTreeHomSpace> {
        &self.homspace
    }

    #[inline]
    pub(crate) fn structure(&self) -> &Arc<BlockStructure> {
        &self.subblock_structure
    }

    pub(crate) fn required_len(&self) -> Result<usize, CoreError> {
        self.subblock_structure.required_len()
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
        .map(|key| (key, vec![1; canonical_rank]))
        .collect::<Vec<_>>();
    let dummy_structure = BlockStructure::coupled_sector_matrix_with_keys(
        rule,
        plan.canonical_dst_nout(),
        canonical_rank,
        dummy_blocks,
    )
    .map_err(OperationError::from_core_preserving_context)?;
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
                if shapes.contains_key(src_tree_key) {
                    merge_inferred_shape(shapes, src_tree_key.clone(), candidate)?;
                }
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

/// Per-axis map from placement-invariant external sector label to degeneracy,
/// collected over all fusion-tree blocks of a structure. Errors if the same
/// (axis, sector) pair appears with two different dims.
fn axis_sector_dims<R>(
    rule: &R,
    structure: &BlockStructure,
) -> Result<Vec<HashMap<SectorId, usize>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let rank = structure.rank();
    let mut dims = vec![HashMap::<SectorId, usize>::new(); rank];
    for index in 0..structure.block_count() {
        let block = structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let BlockKey::FusionTree(key) = block.key() else {
            return Err(OperationError::ExpectedFusionTreeBlock {
                tensor: "scratch source",
                index,
            });
        };
        let sectors = key.external_sectors(rule);
        for (axis, (&sector, &dim)) in sectors.iter().zip(block.shape()).enumerate() {
            match dims[axis].get(&sector) {
                Some(&existing) if existing != dim => {
                    return Err(OperationError::ShapeMismatch {
                        dst: vec![existing],
                        src: vec![dim],
                    });
                }
                Some(_) => {}
                None => {
                    dims[axis].insert(sector, dim);
                }
            }
        }
    }
    Ok(dims)
}
