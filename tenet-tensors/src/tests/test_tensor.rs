//! Rank-erased host operand of the unit-test oracles.
//!
//! Why not a static-rank tensor: the production expert entries take a dynamic
//! space plus a raw slice, so a fixture that carries rank in its type would be
//! a second public-shaped tensor layer kept alive only by tests (#2222).

use std::sync::Arc;

use tenet_core::{
    BlockStructure, CoreError, FusionTensorMapSpace, MultiplicityFreeRigidSymbols, TensorMapSpace,
};

use crate::{DynamicFusionMapSpace, OperationError};

/// Storage plus the layout that describes it: the per-axis dense extents, the
/// block structure, and, for a symmetric operand, the fusion space that owns
/// that structure.
#[derive(Clone, Debug)]
pub(crate) struct TestTensor<T> {
    data: Vec<T>,
    dims: Vec<usize>,
    structure: Arc<BlockStructure>,
    fusion_space: Option<Arc<DynamicFusionMapSpace>>,
}

impl<T> TestTensor<T> {
    /// A dense operand over one trivial block.
    pub(crate) fn from_vec<const NOUT: usize, const NIN: usize>(
        data: Vec<T>,
        space: TensorMapSpace<NOUT, NIN>,
    ) -> Result<Self, CoreError> {
        let structure = BlockStructure::trivial(space.dims())?;
        Self::from_vec_with_structure(data, space, structure)
    }

    /// A dense operand over one trivial block, every element `value`.
    pub(crate) fn filled<const NOUT: usize, const NIN: usize>(
        value: T,
        space: TensorMapSpace<NOUT, NIN>,
    ) -> Result<Self, CoreError>
    where
        T: Clone,
    {
        Self::from_vec(vec![value; space.dense_dim()], space)
    }

    /// An operand over an explicit block layout without a fusion space.
    pub(crate) fn from_vec_with_structure<const NOUT: usize, const NIN: usize>(
        data: Vec<T>,
        space: TensorMapSpace<NOUT, NIN>,
        structure: BlockStructure,
    ) -> Result<Self, CoreError> {
        Self::from_parts(
            data,
            space.dims().to_vec(),
            BlockStructure::canonicalize_shared(structure.into_shared()),
            None,
        )
    }

    /// A symmetric operand; its fusion space owns the block layout.
    pub(crate) fn from_vec_with_fusion_space<const NOUT: usize, const NIN: usize>(
        data: Vec<T>,
        fusion_space: FusionTensorMapSpace<NOUT, NIN>,
    ) -> Result<Self, CoreError> {
        Self::from_vec_with_dynamic_space(
            data,
            fusion_space.dense_space().dims().to_vec(),
            DynamicFusionMapSpace::from_typed(&fusion_space),
        )
    }

    /// A symmetric operand over an already rank-erased fusion space.
    pub(crate) fn from_vec_with_dynamic_space(
        data: Vec<T>,
        dims: Vec<usize>,
        space: DynamicFusionMapSpace,
    ) -> Result<Self, CoreError> {
        let structure = Arc::clone(space.structure());
        Self::from_parts(data, dims, structure, Some(Arc::new(space)))
    }

    fn from_parts(
        data: Vec<T>,
        dims: Vec<usize>,
        structure: Arc<BlockStructure>,
        fusion_space: Option<Arc<DynamicFusionMapSpace>>,
    ) -> Result<Self, CoreError> {
        if structure.rank() != dims.len() {
            return Err(CoreError::StructureRankMismatch {
                expected: dims.len(),
                actual: structure.rank(),
            });
        }
        let required = structure.required_len()?;
        if data.len() != required {
            return Err(CoreError::DimensionMismatch {
                expected: required,
                actual: data.len(),
            });
        }
        Ok(Self {
            data,
            dims,
            structure,
            fusion_space,
        })
    }

    pub(crate) fn data(&self) -> &[T] {
        &self.data
    }

    pub(crate) fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    pub(crate) fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub(crate) fn structure(&self) -> &Arc<BlockStructure> {
        &self.structure
    }

    pub(crate) fn fusion_space(&self) -> Option<&Arc<DynamicFusionMapSpace>> {
        self.fusion_space.as_ref()
    }

    /// The fusion space of a symmetric operand.
    pub(crate) fn dynamic_space(&self) -> &DynamicFusionMapSpace {
        self.fusion_space
            .as_deref()
            .expect("a symmetric fixture carries its fusion space")
    }

    /// The materialized blockwise adjoint, in the coupled-sector layout.
    pub(crate) fn adjoint<R>(&self, rule: &R) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
        T: crate::adjoint::AdjointScalar,
    {
        let space = self.dynamic_space();
        let (adjoint, data) = crate::adjoint::adjoint_dyn(rule, space, &self.data)?;
        let nout = space.nout();
        let dims = [&self.dims[nout..], &self.dims[..nout]].concat();
        Ok(Self::from_vec_with_dynamic_space(data, dims, adjoint)?)
    }
}
