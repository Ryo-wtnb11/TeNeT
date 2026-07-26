use tenet_core::{MultiplicityFreeRigidSymbols, RuleIdentity};
use tenet_tensors::{
    BoundDynamicFusionMapSpace, BoundDynamicTensorRef, FusionOperand, OutputAxisOrder,
    TensorContractSpec, TreeTransformOperation, TreeTransformRuleCacheKey,
};

use crate::runtime::Ctx;
use crate::tensor::UserScalar;

/// Executes one owned multiplicity-free transform from a validated provider
/// binding. The caller keeps user-layer dispatch and representation policy;
/// this helper owns only typed destination derivation and the direct/fallback
/// execution choice.
pub(crate) fn tree_transform_owned_multiplicity_free<R, D>(
    context: &mut Ctx<D, RuleIdentity>,
    input: BoundDynamicTensorRef<'_, R, D>,
    operation: TreeTransformOperation,
) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), tenet_tensors::OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleIdentity>,
    D: UserScalar,
{
    let destination = input.space().transformed_multiplicity_free(&operation)?;
    let dst_space = destination.space();
    if let Some(data) = context
        .tree_context_mut()
        .try_tree_transform_dyn_overwrite_owned(
            input.space().provider(),
            &operation,
            dst_space.structure(),
            input.space().space().structure(),
            dst_space.nout(),
            input.data(),
            D::from_real(1.0),
        )?
    {
        return Ok((destination, data));
    }

    let mut data = vec![D::from_real(0.0); dst_space.required_len()?];
    context.tree_context_mut().tree_transform_dyn_into(
        input.space().provider(),
        operation,
        dst_space.structure(),
        input.space().space().structure(),
        &mut data,
        input.data(),
        D::from_real(1.0),
        D::from_real(0.0),
    )?;
    Ok((destination, data))
}

pub(crate) fn tensorcontract_owned_multiplicity_free<R, D>(
    context: &mut Ctx<D, RuleIdentity>,
    lhs: BoundDynamicTensorRef<'_, R, D>,
    rhs: BoundDynamicTensorRef<'_, R, D>,
    lhs_axes: &[usize],
    rhs_axes: &[usize],
    output_order: OutputAxisOrder<'_>,
) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), tenet_tensors::OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleIdentity>,
    D: UserScalar,
{
    let destination = BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
        lhs.space(),
        rhs.space(),
        lhs_axes,
        rhs_axes,
        output_order,
    )?;
    let mut data = vec![D::from_real(0.0); destination.space().required_len()?];
    context.tensorcontract_fusion_dyn_into(
        &destination,
        &mut data,
        lhs.space(),
        lhs.data(),
        rhs.space(),
        rhs.data(),
        TensorContractSpec::new(lhs_axes, rhs_axes, output_order),
        D::from_real(1.0),
        D::from_real(0.0),
    )?;
    Ok((destination, data))
}

/// Categorical composition (TensorKit `A * B` / `mul!`) of two owned
/// multiplicity-free operands.
///
/// Differs from [`tensorcontract_owned_multiplicity_free`] in exactly two
/// places: the output order is fixed to the identity (composition has no
/// re-ordering freedom — the open axes keep their sides), and the seam is the
/// composition one, which never inserts the fermionic supertrace twist.
///
/// The operands are always direct: this facade has no lazy adjoint view, so
/// there is no conjugated storage for [`tenet_tensors::FusionOperand`] to
/// separate from its logical geometry.
pub(crate) fn tensorcompose_owned_multiplicity_free<R, D>(
    context: &mut Ctx<D, RuleIdentity>,
    lhs: BoundDynamicTensorRef<'_, R, D>,
    rhs: BoundDynamicTensorRef<'_, R, D>,
    lhs_axes: &[usize],
    rhs_axes: &[usize],
) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), tenet_tensors::OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleIdentity>,
    D: UserScalar,
{
    let destination = BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
        lhs.space(),
        rhs.space(),
        lhs_axes,
        rhs_axes,
        OutputAxisOrder::identity(),
    )?;
    let mut data = vec![D::from_real(0.0); destination.space().required_len()?];
    context.tensorcompose_fusion_dyn_into(
        &destination,
        &mut data,
        FusionOperand::direct(lhs.space().space()),
        lhs.data(),
        FusionOperand::direct(rhs.space().space()),
        rhs.data(),
        lhs_axes,
        rhs_axes,
        D::from_real(1.0),
        D::from_real(0.0),
    )?;
    Ok((destination, data))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tenet_core::{
        BraidingStyleKind, FusionProductSpace, FusionRule, FusionStyleKind,
        MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols,
        RuleIdentity, SectorId, SectorLeg, SectorVec, Z2FusionRule,
    };
    use tenet_tensors::{
        BoundDynamicFusionMapSpace, BoundDynamicTensorRef, OutputAxisOrder, TreeTransformOperation,
    };

    use super::{tensorcontract_owned_multiplicity_free, tree_transform_owned_multiplicity_free};
    use crate::runtime::Ctx;

    /// Deliberately outside the user-layer rule enum: this exercises the typed
    /// core with a provider an application can define without `LoweredMultiplicityFreeAlgebra`.
    struct ExternalZ2;

    impl FusionRule for ExternalZ2 {
        fn rule_identity(&self) -> RuleIdentity {
            RuleIdentity::of_type::<Self>()
        }

        fn fusion_style(&self) -> FusionStyleKind {
            Z2FusionRule.fusion_style()
        }

        fn braiding_style(&self) -> BraidingStyleKind {
            Z2FusionRule.braiding_style()
        }

        fn vacuum(&self) -> SectorId {
            Z2FusionRule.vacuum()
        }

        fn supports_unitary_braid_dagger(&self) -> bool {
            Z2FusionRule.supports_unitary_braid_dagger()
        }

        fn dual(&self, sector: SectorId) -> SectorId {
            Z2FusionRule.dual(sector)
        }

        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            Z2FusionRule.fusion_channels(left, right)
        }
    }

    impl MultiplicityFreeFusionRule for ExternalZ2 {}

    impl MultiplicityFreeFusionSymbols for ExternalZ2 {
        type Scalar = f64;

        fn scalar_one(&self) -> Self::Scalar {
            Z2FusionRule.scalar_one()
        }

        fn scalar_conj(&self, value: Self::Scalar) -> Self::Scalar {
            Z2FusionRule.scalar_conj(value)
        }

        fn has_trivial_associator_gauge(&self) -> bool {
            Z2FusionRule.has_trivial_associator_gauge()
        }

        fn f_symbol_scalar(
            &self,
            left: SectorId,
            middle: SectorId,
            right: SectorId,
            coupled: SectorId,
            left_coupled: SectorId,
            right_coupled: SectorId,
        ) -> Self::Scalar {
            Z2FusionRule.f_symbol_scalar(left, middle, right, coupled, left_coupled, right_coupled)
        }

        fn r_symbol_scalar(
            &self,
            left: SectorId,
            right: SectorId,
            coupled: SectorId,
        ) -> Self::Scalar {
            Z2FusionRule.r_symbol_scalar(left, right, coupled)
        }
    }

    impl MultiplicityFreeRigidSymbols for ExternalZ2 {
        fn dim_scalar(&self, sector: SectorId) -> Self::Scalar {
            Z2FusionRule.dim_scalar(sector)
        }

        fn inv_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
            Z2FusionRule.inv_dim_scalar(sector)
        }

        fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
            Z2FusionRule.sqrt_dim_scalar(sector)
        }

        fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
            Z2FusionRule.inv_sqrt_dim_scalar(sector)
        }

        fn twist_scalar(&self, sector: SectorId) -> Self::Scalar {
            Z2FusionRule.twist_scalar(sector)
        }

        fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> Self::Scalar {
            Z2FusionRule.frobenius_schur_phase_scalar(sector)
        }
    }

    #[test]
    fn external_multiplicity_free_provider_matches_direct_transform() {
        let provider = Arc::new(ExternalZ2);
        let leg = SectorLeg::new([(SectorId::new(0), 2)], false);
        let source = BoundDynamicFusionMapSpace::from_degeneracy_shapes(
            Arc::clone(&provider),
            tenet_core::FusionTreeHomSpace::new(
                FusionProductSpace::new([leg.clone()]),
                FusionProductSpace::new([leg]),
            ),
            [vec![2, 2]],
        )
        .unwrap();
        let operation = TreeTransformOperation::permute([1], [0]);
        let source_data = vec![1.0, 2.0, 3.0, 4.0];

        let expected_destination = source.transformed_multiplicity_free(&operation).unwrap();
        let mut expected_data = vec![0.0; expected_destination.space().required_len().unwrap()];
        let mut direct = Ctx::<f64, RuleIdentity>::default();
        direct
            .tree_context_mut()
            .tree_transform_dyn_into(
                provider.as_ref(),
                operation.clone(),
                expected_destination.space().structure(),
                source.space().structure(),
                &mut expected_data,
                &source_data,
                1.0,
                0.0,
            )
            .unwrap();

        let input: tenet_matrixalgebra::BoundDynamicTensorRef<'_, ExternalZ2, f64> =
            BoundDynamicTensorRef::try_new(&source, &source_data).unwrap();
        let mut context = Ctx::<f64, RuleIdentity>::default();
        let (actual_destination, actual_data) =
            tree_transform_owned_multiplicity_free(&mut context, input, operation).unwrap();

        assert_eq!(actual_destination.space(), expected_destination.space());
        assert_eq!(actual_data, expected_data);
    }

    #[test]
    fn external_multiplicity_free_provider_contracts_direct_with_output_order() {
        let provider = Arc::new(ExternalZ2);
        let lhs_codomain = SectorLeg::new([(SectorId::new(0), 2)], false);
        let contracted = SectorLeg::new([(SectorId::new(0), 3)], false);
        let rhs_domain = SectorLeg::new([(SectorId::new(0), 4)], false);
        let lhs = BoundDynamicFusionMapSpace::from_degeneracy_shapes(
            Arc::clone(&provider),
            tenet_core::FusionTreeHomSpace::new(
                FusionProductSpace::new([lhs_codomain.clone()]),
                FusionProductSpace::new([contracted.clone()]),
            ),
            [vec![2, 3]],
        )
        .unwrap();
        let rhs = BoundDynamicFusionMapSpace::from_degeneracy_shapes(
            Arc::clone(&provider),
            tenet_core::FusionTreeHomSpace::new(
                FusionProductSpace::new([contracted]),
                FusionProductSpace::new([rhs_domain.clone()]),
            ),
            [vec![3, 4]],
        )
        .unwrap();
        let lhs_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rhs_data = vec![
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        ];

        let lhs = BoundDynamicTensorRef::try_new(&lhs, &lhs_data).unwrap();
        let rhs = BoundDynamicTensorRef::try_new(&rhs, &rhs_data).unwrap();
        let mut context = Ctx::<f64, RuleIdentity>::default();
        let (destination, data) = tensorcontract_owned_multiplicity_free(
            &mut context,
            lhs,
            rhs,
            &[1],
            &[0],
            OutputAxisOrder::from_axes(&[1, 0]),
        )
        .unwrap();

        let expected_destination = tenet_core::FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(SectorId::new(0), 4)], true)]),
            FusionProductSpace::new([SectorLeg::new([(SectorId::new(0), 2)], true)]),
        );
        assert_eq!(destination.space().homspace(), &expected_destination);
        assert_eq!(
            data,
            [76.0, 103.0, 130.0, 157.0, 100.0, 136.0, 172.0, 208.0]
        );
    }
}
