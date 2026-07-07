use std::sync::{Arc, OnceLock, RwLock};

use rustc_hash::FxHashMap;
use tenet_core::{
    BlockKey, BlockStructure, CoreError, FusionTensorMapSpace, FusionTreeHomSpace,
    MultiplicityFreeRigidSymbols,
};

use crate::{OperationError, TreeTransformOperation};
use tenet_operations::TensorContractSpec;

/// Identity of a pairwise contraction's output space: the two operand hom
/// spaces (by value — `Arc` gives cheap clones while `Hash`/`Eq` delegate to
/// the pointee, so a rebuilt-but-identical space still matches) plus the
/// contracted axis lists. The hom spaces carry the full sector/leg structure
/// (authoritative even for structural-zero keys the subblock layout omits, so
/// a subblock content id alone is NOT enough), and the output is a pure
/// function of these. The same contraction across sweeps/evals thus reuses one
/// built space instead of rebuilding the coupled-sector layout each call.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct ContractedSpaceKey {
    /// Fusion rule discriminant: the same sector ids fuse differently under
    /// different rules, so the process-global cache must key on the rule.
    /// Mirrors `FusionTreeHomSpaceCacheKey` in tenet-core.
    rule_type: &'static str,
    lhs_homspace: Arc<FusionTreeHomSpace>,
    rhs_homspace: Arc<FusionTreeHomSpace>,
    lhs_axes: Vec<usize>,
    rhs_axes: Vec<usize>,
}

fn contracted_space_cache() -> &'static RwLock<FxHashMap<ContractedSpaceKey, DynamicFusionMapSpace>>
{
    static CACHE: OnceLock<RwLock<FxHashMap<ContractedSpaceKey, DynamicFusionMapSpace>>> =
        OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(FxHashMap::default()))
}

/// Identity of a tree-transform (permute / braid / transpose) output space:
/// the source hom space (by value — the legs carry the authoritative
/// sector→degeneracy map the output shapes derive from) plus the transform
/// operation, under a given fusion rule. Mirrors [`ContractedSpaceKey`].
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct TransformedSpaceKey {
    rule_type: &'static str,
    source_homspace: Arc<FusionTreeHomSpace>,
    operation: TreeTransformOperation,
}

fn transformed_space_cache(
) -> &'static RwLock<FxHashMap<TransformedSpaceKey, DynamicFusionMapSpace>> {
    static CACHE: OnceLock<RwLock<FxHashMap<TransformedSpaceKey, DynamicFusionMapSpace>>> =
        OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(FxHashMap::default()))
}

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

use super::fusion::FusionContractPlan;
use super::structure::TensorContractAxisPlan;

/// Dynamic-rank fusion space: the expert-layer space handle whose
/// codomain/domain split is a runtime property.
///
/// Typed [`FusionTensorMapSpace`] facades lower to this type internally; the
/// dynamic expert entry points (`*_dyn_into`) take it directly together with
/// raw `f64` slices in the coupled-sector matrix layout.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DynamicFusionMapSpace {
    nout: usize,
    nin: usize,
    homspace: Arc<FusionTreeHomSpace>,
    subblock_structure: Arc<BlockStructure>,
}

impl DynamicFusionMapSpace {
    /// Rank-erases a typed fusion space (shares the hom space and subblock
    /// structure handles; no data copies).
    pub fn from_typed<const NOUT: usize, const NIN: usize>(
        space: &FusionTensorMapSpace<NOUT, NIN>,
    ) -> Self {
        Self {
            nout: NOUT,
            nin: NIN,
            homspace: Arc::clone(space.homspace_arc()),
            subblock_structure: Arc::clone(space.subblock_structure()),
        }
    }

    /// Builds a dynamic space directly from an untyped description: a hom
    /// space plus one degeneracy shape per fusion-tree key (in
    /// [`FusionTreeHomSpace::fusion_tree_keys`] order). The storage layout is
    /// the TensorKit-equivalent coupled-sector matrix layout, identical to
    /// [`FusionTensorMapSpace::from_degeneracy_shapes`].
    pub fn from_degeneracy_shapes<R, Shapes>(
        rule: &R,
        homspace: FusionTreeHomSpace,
        shapes: Shapes,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
        Shapes: IntoIterator,
        Shapes::Item: Into<Vec<usize>>,
    {
        let nout = homspace.codomain().len();
        let nin = homspace.domain().len();
        let keys = homspace.fusion_tree_keys(rule);
        let shapes = shapes.into_iter().map(Into::into).collect::<Vec<_>>();
        if keys.len() != shapes.len() {
            return Err(OperationError::from_core_preserving_context(
                CoreError::BlockCountMismatch {
                    expected: keys.len(),
                    actual: shapes.len(),
                },
            ));
        }
        homspace
            .validate_degeneracy_shapes(&keys, &shapes)
            .map_err(OperationError::from_core_preserving_context)?;
        let blocks = keys
            .into_iter()
            .map(BlockKey::from)
            .zip(shapes)
            .collect::<Vec<_>>();
        let subblock_structure =
            Arc::new(scratch_subblock_structure(rule, nout, nout + nin, blocks)?);
        Ok(Self {
            nout,
            nin,
            homspace: Arc::new(homspace),
            subblock_structure,
        })
    }

    pub(crate) fn transformed_from_typed<R, const NOUT: usize, const NIN: usize>(
        rule: &R,
        source: &FusionTensorMapSpace<NOUT, NIN>,
        operation: &TreeTransformOperation,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        Self::from_typed(source).transformed(rule, operation)
    }

    /// Space of the tree-transformed (permute / braid / transpose) tensor:
    /// the hom space is permuted and the full tree set of the result is
    /// enumerated (trees the transform coefficients never reach stay as
    /// structural zeros, keeping every coupled sector grid complete).
    pub fn transformed<R>(
        &self,
        rule: &R,
        operation: &TreeTransformOperation,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        let source = self;
        let cache_key = TransformedSpaceKey {
            rule_type: std::any::type_name::<R>(),
            source_homspace: Arc::clone(&source.homspace),
            operation: operation.clone(),
        };
        if let Some(cached) = transformed_space_cache()
            .read()
            .ok()
            .and_then(|map| map.get(&cache_key).cloned())
        {
            return Ok(cached);
        }
        let (codomain_axes, domain_axes) = tree_transform_operation_axes(operation);
        let nout = codomain_axes.len();
        let nin = domain_axes.len();
        let homspace = source
            .homspace()
            .permute(rule, codomain_axes, domain_axes)
            .map_err(OperationError::from_core_preserving_context)?;
        let src_axes = codomain_axes
            .iter()
            .chain(domain_axes.iter())
            .copied()
            .collect::<Vec<_>>();
        // Legs are authoritative for degeneracies: the external leg of each
        // source axis carries the full sector -> degeneracy map, keyed by
        // the placement-invariant external sector labels.
        let src_legs = src_axes
            .iter()
            .map(|&src_axis| source.homspace().external_axis_leg(rule, src_axis))
            .collect::<Vec<_>>();
        let keys = homspace.fusion_tree_keys(rule);
        let mut blocks = Vec::<(BlockKey, Vec<usize>)>::with_capacity(keys.len());
        for key in keys {
            let sectors = key.external_sectors(rule);
            let mut shape = Vec::with_capacity(src_axes.len());
            for (out_axis, leg) in src_legs.iter().enumerate() {
                let dim =
                    leg.degeneracy(sectors[out_axis])
                        .ok_or(OperationError::StructureMismatch {
                            tensor: "transformed scratch",
                        })?;
                shape.push(dim);
            }
            blocks.push((BlockKey::from(key), shape));
        }
        let subblock_structure =
            Arc::new(scratch_subblock_structure(rule, nout, nout + nin, blocks)?);
        let space = Self {
            nout,
            nin,
            homspace: Arc::new(homspace),
            subblock_structure,
        };
        if let Ok(mut map) = transformed_space_cache().write() {
            map.insert(cache_key, space.clone());
        }
        Ok(space)
    }

    /// Space of the contraction result in the default output order (`lhs`
    /// open axes ascending on the codomain side, `rhs` open axes ascending on
    /// the domain side). Mirrors the destination TensorKit's
    /// `tensorcontract!` with default `pAB` writes into.
    pub fn contracted<R>(
        rule: &R,
        lhs: &Self,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        if lhs_axes.len() != rhs_axes.len() {
            return Err(OperationError::ContractAxisCountMismatch {
                lhs: lhs_axes.len(),
                rhs: rhs_axes.len(),
            });
        }
        let nout = lhs
            .rank()
            .checked_sub(lhs_axes.len())
            .ok_or(OperationError::RankMismatch {
                expected: lhs_axes.len(),
                actual: lhs.rank(),
            })?;
        let nin = rhs
            .rank()
            .checked_sub(rhs_axes.len())
            .ok_or(OperationError::RankMismatch {
                expected: rhs_axes.len(),
                actual: rhs.rank(),
            })?;
        let key = ContractedSpaceKey {
            rule_type: std::any::type_name::<R>(),
            lhs_homspace: Arc::clone(&lhs.homspace),
            rhs_homspace: Arc::clone(&rhs.homspace),
            lhs_axes: lhs_axes.to_vec(),
            rhs_axes: rhs_axes.to_vec(),
        };
        if let Some(cached) = contracted_space_cache()
            .read()
            .ok()
            .and_then(|map| map.get(&key).cloned())
        {
            return Ok(cached);
        }
        let axes = TensorContractSpec::with_default_output_order(lhs_axes, rhs_axes);
        let space = Self::contracted_space(rule, lhs, rhs, axes, nout, nin)?;
        if let Ok(mut map) = contracted_space_cache().write() {
            map.insert(key, space.clone());
        }
        Ok(space)
    }

    pub(crate) fn core_dst<R>(
        rule: &R,
        lhs: &Self,
        rhs: &Self,
        plan: &FusionContractPlan,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        let nout = plan.core_dst_open_lhs_rank();
        let nin = plan.core_dst_open_rhs_rank();
        Self::contracted_space(rule, lhs, rhs, plan.core_axes().as_spec(), nout, nin)
    }

    fn contracted_space<R>(
        rule: &R,
        lhs: &Self,
        rhs: &Self,
        axes: TensorContractSpec<'_>,
        nout: usize,
        nin: usize,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
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

        // The legs are authoritative for every subblock shape: a fusion-tree
        // key's shape is fully determined by each open axis' leg degeneracy
        // at the key's external sector (TensorKit GradedSpace parity: the
        // legs carry the complete sector -> degeneracy map, so shapes are
        // recoverable even for structural-zero keys the contraction pairing
        // never produces, e.g. sparse product states or factors of a
        // truncated SVD that dropped a whole coupled sector).
        let lhs_open = axis_plan.lhs_open_axes.clone();
        let rhs_open = axis_plan.rhs_open_axes.clone();
        let open_legs = lhs_open
            .iter()
            .map(|&axis| lhs.homspace().external_axis_leg(rule, axis))
            .chain(
                rhs_open
                    .iter()
                    .map(|&axis| rhs.homspace().external_axis_leg(rule, axis)),
            )
            .collect::<Vec<_>>();
        let keys = homspace.fusion_tree_keys(rule);
        let mut blocks = Vec::<(BlockKey, Vec<usize>)>::with_capacity(keys.len());
        for key in keys {
            let sectors = key.external_sectors(rule);
            let shape = sectors
                .iter()
                .zip(&open_legs)
                .map(|(&sector, leg)| {
                    leg.degeneracy(sector)
                        .ok_or(OperationError::StructureMismatch {
                            tensor: "contracted result",
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
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

    /// Adjoint view: codomain and domain swap (spaces and per-block shapes),
    /// no data movement implied. The block layout is a strided view into the
    /// source layout, so this space is for replay bookkeeping, not for
    /// allocating fresh coupled storage.
    pub(crate) fn adjoint_view(&self) -> Result<Self, OperationError> {
        let homspace = FusionTreeHomSpace::new(
            self.homspace.domain().clone(),
            self.homspace.codomain().clone(),
        );
        let structure = crate::lowering::adjoint_block_structure_view(
            self.nout,
            self.nin,
            &self.subblock_structure,
        )?;
        Ok(Self {
            nout: self.nin,
            nin: self.nout,
            homspace: Arc::new(homspace),
            subblock_structure: Arc::new(structure),
        })
    }

    /// Number of codomain legs.
    #[inline]
    pub fn nout(&self) -> usize {
        self.nout
    }

    /// Number of domain legs.
    #[inline]
    pub fn nin(&self) -> usize {
        self.nin
    }

    /// Total number of legs.
    #[inline]
    pub fn rank(&self) -> usize {
        self.nout + self.nin
    }

    #[inline]
    pub fn homspace(&self) -> &FusionTreeHomSpace {
        &self.homspace
    }

    /// Shared hom-space handle for pointer-identity fast paths in replay
    /// caches.
    pub fn homspace_arc(&self) -> &Arc<FusionTreeHomSpace> {
        &self.homspace
    }

    /// Subblock (fusion-tree) block structure of the coupled storage layout.
    #[inline]
    pub fn structure(&self) -> &Arc<BlockStructure> {
        &self.subblock_structure
    }

    /// Flat storage length this space requires.
    pub fn required_len(&self) -> Result<usize, CoreError> {
        self.subblock_structure.required_len()
    }
}

fn tree_transform_operation_axes(operation: &TreeTransformOperation) -> (&[usize], &[usize]) {
    match operation {
        TreeTransformOperation::Transpose {
            codomain_permutation,
            domain_permutation,
        }
        | TreeTransformOperation::Permute {
            codomain_permutation,
            domain_permutation,
        }
        | TreeTransformOperation::Braid {
            codomain_permutation,
            domain_permutation,
            ..
        } => (
            codomain_permutation.as_slice(),
            domain_permutation.as_slice(),
        ),
    }
}
