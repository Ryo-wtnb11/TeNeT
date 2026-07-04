//! Composition of factor maps with automatic destination allocation.

use std::hash::Hash;

use tenet_core::{
    BlockKey, CoreError, FusionTensorMapSpace, FusionTreeHomSpace, MultiplicityFreeRigidSymbols,
    TensorMap, TensorMapSpace,
};
use tenet_tensors::{
    OperationError, OutputAxisOrder, TensorContractFusionExecutionContext, TensorContractSpec,
    TreeTransformRuleCacheKey,
};

use crate::factorize::FactorScalar;

/// `lhs . rhs` over the full domain/codomain interface, allocating the
/// destination in the coupled layout. The recomposition step of the derived
/// operations (`V f(D) V^H`, `U Vh`, ...).
pub(crate) fn compose<RuleKey, BT, BC, R, D, const A: usize, const B: usize, const C: usize>(
    context: &mut TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    rule: &R,
    lhs: &TensorMap<D, A, B>,
    rhs: &TensorMap<D, B, C>,
) -> Result<TensorMap<D, A, C>, OperationError>
where
    RuleKey: Clone + Eq + Hash,
    BT: tenet_tensors::TreeTransformBackend<D, f64>,
    BC: tenet_tensors::TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    let lhs_space = lhs
        .fusion_space()
        .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?;
    let rhs_space = rhs
        .fusion_space()
        .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?;

    let lhs_axes: Vec<usize> = (A..A + B).collect();
    let rhs_axes: Vec<usize> = (0..B).collect();
    let output_axes: Vec<usize> = (0..A + C).collect();
    let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
        rule,
        lhs_space.homspace(),
        rhs_space.homspace(),
        &lhs_axes,
        &rhs_axes,
        &output_axes,
        A,
    )
    .map_err(OperationError::from_core_preserving_context)?;

    let lhs_structure = std::sync::Arc::clone(lhs.structure());
    let rhs_structure = std::sync::Arc::clone(rhs.structure());
    let shape_of_side = |structure: &tenet_core::BlockStructure,
                         tree: &tenet_core::FusionTreeKey,
                         codomain_side: bool,
                         nout: usize|
     -> Option<Vec<usize>> {
        for index in 0..structure.block_count() {
            let block = structure.block(index).ok()?;
            let BlockKey::FusionTree(key) = block.key() else {
                continue;
            };
            let candidate = if codomain_side {
                key.codomain_tree()
            } else {
                key.domain_tree()
            };
            if candidate == tree {
                return Some(if codomain_side {
                    block.shape()[..nout].to_vec()
                } else {
                    block.shape()[nout..].to_vec()
                });
            }
        }
        None
    };

    let keys = dst_hom.fusion_tree_keys(rule);
    let shapes = keys
        .iter()
        .map(|key| {
            let mut shape = shape_of_side(&lhs_structure, key.codomain_tree(), true, A).ok_or(
                OperationError::UnsupportedTensorContractScope {
                    message: "composition codomain tree absent from the left factor",
                },
            )?;
            shape.extend(
                shape_of_side(&rhs_structure, key.domain_tree(), false, B).ok_or(
                    OperationError::UnsupportedTensorContractScope {
                        message: "composition domain tree absent from the right factor",
                    },
                )?,
            );
            Ok(shape)
        })
        .collect::<Result<Vec<_>, OperationError>>()?;

    let lhs_dims = lhs.space().dims();
    let rhs_dims = rhs.space().dims();
    let mut codomain_dims = [0usize; A];
    codomain_dims.copy_from_slice(&lhs_dims[..A]);
    let mut domain_dims = [0usize; C];
    domain_dims.copy_from_slice(&rhs_dims[B..]);
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<A, C>::from_dims(codomain_dims, domain_dims)
            .map_err(OperationError::from_core_preserving_context)?,
        dst_hom,
        rule,
        shapes,
    )
    .map_err(OperationError::from_core_preserving_context)?;
    let len = space
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut dst = TensorMap::<D, A, C>::from_vec_with_fusion_space(vec![D::zero(); len], space)
        .map_err(OperationError::from_core_preserving_context)?;

    let perm: Vec<usize> = (0..A + C).collect();
    context.tensorcontract_fusion_into(
        rule,
        &mut dst,
        lhs,
        rhs,
        TensorContractSpec::new(&lhs_axes, &rhs_axes, OutputAxisOrder::from_axes(&perm)),
        D::one(),
        D::zero(),
    )?;
    Ok(dst)
}
