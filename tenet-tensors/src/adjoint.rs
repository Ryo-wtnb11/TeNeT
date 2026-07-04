//! Adjoint (dagger) of a fusion tensor.
//!
//! TensorKit semantics: the adjoint of `t : codomain <- domain` is
//! `t^H : domain <- codomain`, whose coupled-sector blocks are the conjugate
//! transposes of `t`'s blocks (`block(t^H, c) = block(t, c)^H`). Codomain and
//! domain swap as spaces; leg duality flags are unchanged.

use std::sync::Arc;

use tenet_core::{
    BlockKey, CoreError, FusionTensorMapSpace, FusionTreeBlockKey, FusionTreeHomSpace,
    MultiplicityFreeRigidSymbols, TensorMap, TensorMapSpace,
};

use crate::contract::DynamicFusionMapSpace;
use crate::{ConjugateValue, OperationError};

/// Dynamic-rank adjoint (dagger): returns the adjoint space (codomain and
/// domain swapped) together with freshly allocated coupled-layout data whose
/// blocks are the conjugate transposes of the source blocks.
pub fn adjoint_dyn<R, D>(
    rule: &R,
    space: &DynamicFusionMapSpace,
    data: &[D],
) -> Result<(DynamicFusionMapSpace, Vec<D>), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: Copy + num_traits::Zero + Clone + ConjugateValue,
{
    let nout = space.nout();
    let nin = space.nin();
    let homspace = space.homspace();
    let adjoint_hom =
        FusionTreeHomSpace::new(homspace.domain().clone(), homspace.codomain().clone());

    let structure = Arc::clone(space.structure());
    let keys = adjoint_hom.fusion_tree_keys(rule);
    let shapes = keys
        .iter()
        .map(|key| {
            let source_key = BlockKey::FusionTree(FusionTreeBlockKey::pair(
                key.domain_tree().clone(),
                key.codomain_tree().clone(),
            ));
            let index = structure.find_block_index_by_key(&source_key).ok_or(
                OperationError::MissingBlockKey {
                    key: source_key.clone(),
                },
            )?;
            let source_shape = structure
                .block(index)
                .map_err(OperationError::from_core_preserving_context)?
                .shape();
            let mut shape = source_shape[nout..].to_vec();
            shape.extend_from_slice(&source_shape[..nout]);
            Ok(shape)
        })
        .collect::<Result<Vec<_>, OperationError>>()?;

    let adjoint_space = DynamicFusionMapSpace::from_degeneracy_shapes(rule, adjoint_hom, shapes)?;
    let len = adjoint_space
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut result = vec![D::zero(); len];

    let result_structure = Arc::clone(adjoint_space.structure());
    for index in 0..result_structure.block_count() {
        let block = result_structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        let source_key = BlockKey::FusionTree(FusionTreeBlockKey::pair(
            key.domain_tree().clone(),
            key.codomain_tree().clone(),
        ));
        let source_index = structure
            .find_block_index_by_key(&source_key)
            .ok_or(OperationError::MissingBlockKey { key: source_key })?;
        let source_block = structure
            .block(source_index)
            .map_err(OperationError::from_core_preserving_context)?;

        let shape = block.shape().to_vec();
        let strides = block.strides().to_vec();
        let offset = block.offset();
        let source_strides = source_block.strides().to_vec();
        let source_offset = source_block.offset();
        // Adjoint index map: result (j[..nin], i[..nout]) reads
        // conj(source(i, j)).
        let count: usize = shape.iter().product();
        let mut indices = vec![0usize; shape.len()];
        for _ in 0..count {
            let position = offset
                + indices
                    .iter()
                    .zip(&strides)
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>();
            let source_position = source_offset
                + indices[nin..]
                    .iter()
                    .zip(&source_strides[..nout])
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>()
                + indices[..nin]
                    .iter()
                    .zip(&source_strides[nout..])
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>();
            result[position] = data[source_position].maybe_conj(true);
            for axis in 0..shape.len() {
                indices[axis] += 1;
                if indices[axis] < shape[axis] {
                    break;
                }
                indices[axis] = 0;
            }
        }
    }
    Ok((adjoint_space, result))
}

/// Eager blockwise adjoint; the output uses the coupled-sector matrix layout.
pub fn adjoint<R, D, const NOUT: usize, const NIN: usize>(
    rule: &R,
    tensor: &TensorMap<D, NOUT, NIN>,
) -> Result<TensorMap<D, NIN, NOUT>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: Copy + num_traits::Zero + Clone + ConjugateValue,
{
    let fusion_space = tensor
        .fusion_space()
        .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?;
    let homspace = fusion_space.homspace();
    let adjoint_hom =
        FusionTreeHomSpace::new(homspace.domain().clone(), homspace.codomain().clone());

    let structure = Arc::clone(tensor.structure());
    let source_shape_of = |key: &FusionTreeBlockKey| -> Result<Vec<usize>, OperationError> {
        // The adjoint block for (dom_tree, cod_tree) reads the source block
        // keyed by the swapped pair.
        let source_key = BlockKey::FusionTree(FusionTreeBlockKey::pair(
            key.domain_tree().clone(),
            key.codomain_tree().clone(),
        ));
        let index = structure.find_block_index_by_key(&source_key).ok_or(
            OperationError::MissingBlockKey {
                key: source_key.clone(),
            },
        )?;
        Ok(structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?
            .shape()
            .to_vec())
    };

    let keys = adjoint_hom.fusion_tree_keys(rule);
    let shapes = keys
        .iter()
        .map(|key| {
            let source_shape = source_shape_of(key)?;
            let mut shape = source_shape[NOUT..].to_vec();
            shape.extend_from_slice(&source_shape[..NOUT]);
            Ok(shape)
        })
        .collect::<Result<Vec<_>, OperationError>>()?;

    let dims = tensor.space().dims();
    let mut domain_dims = [0usize; NIN];
    domain_dims.copy_from_slice(&dims[NOUT..]);
    let mut codomain_dims = [0usize; NOUT];
    codomain_dims.copy_from_slice(&dims[..NOUT]);
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<NIN, NOUT>::from_dims(domain_dims, codomain_dims)
            .map_err(OperationError::from_core_preserving_context)?,
        adjoint_hom,
        rule,
        shapes,
    )
    .map_err(OperationError::from_core_preserving_context)?;
    let len = space
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut result =
        TensorMap::<D, NIN, NOUT>::from_vec_with_fusion_space(vec![D::zero(); len], space)
            .map_err(OperationError::from_core_preserving_context)?;

    let result_structure = Arc::clone(result.structure());
    let source_data = tensor.data();
    for index in 0..result_structure.block_count() {
        let block = result_structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        let source_key = BlockKey::FusionTree(FusionTreeBlockKey::pair(
            key.domain_tree().clone(),
            key.codomain_tree().clone(),
        ));
        let source_index = structure
            .find_block_index_by_key(&source_key)
            .ok_or(OperationError::MissingBlockKey { key: source_key })?;
        let source_block = structure
            .block(source_index)
            .map_err(OperationError::from_core_preserving_context)?;

        let shape = block.shape().to_vec();
        let strides = block.strides().to_vec();
        let offset = block.offset();
        let source_strides = source_block.strides().to_vec();
        let source_offset = source_block.offset();
        // Adjoint index map: result (j[..NIN], i[..NOUT]) reads
        // conj(source(i, j)).
        let count: usize = shape.iter().product();
        let mut indices = vec![0usize; shape.len()];
        let data = result.data_mut();
        for _ in 0..count {
            let position = offset
                + indices
                    .iter()
                    .zip(&strides)
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>();
            let source_position = source_offset
                + indices[NIN..]
                    .iter()
                    .zip(&source_strides[..NOUT])
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>()
                + indices[..NIN]
                    .iter()
                    .zip(&source_strides[NOUT..])
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>();
            data[position] = source_data[source_position].maybe_conj(true);
            for axis in 0..shape.len() {
                indices[axis] += 1;
                if indices[axis] < shape[axis] {
                    break;
                }
                indices[axis] = 0;
            }
        }
    }
    Ok(result)
}
