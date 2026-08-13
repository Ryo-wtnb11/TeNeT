# Expert facade migration (pre-release)

> **Historical migration manifest.** Snapshot authority: TeNeT
> `aa7afbc70ced74c17cdb5d42533c4ceda1c913f0`. This is not a current export
> inventory; see the [historical-record index](history.md).

TeNeT is pre-release (`0.1.0`, with no published TeNeT release), so the
facade cleanup may remove paths. Runtime behavior is unchanged.

The following manifest is the former glob surface minus the v2 curated
allow-list. Every listed `tenet::operations::NAME` moves unchanged to
`tenet_tensors::NAME`; every listed `tenet::matrixalgebra::NAME` moves
unchanged to `tenet_matrixalgebra::NAME`. It is reconciled against the two
implementation-crate `lib.rs` re-export lists, including conditional and
`doc(hidden)` re-exports.

### `tenet::operations` → `tenet_tensors`

- `RuleIdentity`; `adjoint`, `adjoint_bound_dyn`, `adjoint_bound_dyn_generic`,
  `adjoint_bound_space_dyn`, `adjoint_bound_space_dyn_generic`;
  `TensorTraceOperationsBackend`; `BoundDynamicTensorRef`.
- `reset_global_operation_caches`, `BlockStructureCacheBlockKey`,
  `BlockStructureCacheKey`, `OperationCachePolicy`,
  `TensorContractStructureCache`, `TensorContractStructureCacheKey`,
  `TreeTransformStructureCache`, `TreeTransformStructureCacheKey`.
- `prepare_tensorcontract_fusion_plan`, `prepare_tensorcontract_fusion_plan_dyn`,
  `tensorcontract_execute_with`, `tensorcontract_fusion_block_specs`,
  `tensorcontract_fusion_into_with`, `tensorcontract_fusion_into_with_backends`,
  `tensorcontract_fusion_prepared_into`,
  `tensorcontract_fusion_prepared_into_core_dst`,
  `tensorcontract_fusion_prepared_into_core_dst_with`,
  `tensorcontract_fusion_prepared_into_with`, `tensorcontract_fusion_structure`,
  `tensorcontract_fusion_structure_dyn`,
  `tensorcontract_fusion_via_tree_pair_transforms_into`,
  `tensorcontract_into_with`, `tensorcontract_into_with_context`,
  `tensorcontract_structure`, `tensorproduct_fusion_into`,
  `tensorproduct_fusion_into_with_conjugation`, `tensorproduct_into`,
  `tensorproduct_into_with_conjugation`, `FusionContractPlan`,
  `HostTensorContractBackend`, `HostTensorContractWorkspace`,
  `HostTreeFusionExecutionContext`, `PreparedTensorContractFusion`,
  `TensorContractBackend`, `TensorContractBlockSpec`, `TensorContractCache`,
  `TensorContractCacheStats`, `TensorContractExecutionContext`,
  `TensorContractFusionProfile`, `TensorContractFusionRoute`,
  `TensorContractPlanKey`, `TensorContractStructure`,
  `TensorContractStructureTerm`, `TensorContractWorkspace`, `FusionOperand`,
  `ValidatedDynamicFusionLayout`.
- `braid_into_with`, `braid_into_with_context`, `permute_into_with`,
  `permute_into_with_context`, `transpose_into_with`,
  `transpose_into_with_context`, `braid_into_generic`, `permute_into_generic`,
  `transpose_into_generic`, `tree_transform_into_generic`,
  `tree_transform_into_with_generic`, `tree_transform_structure_generic`.
- `copy_into`, `scaled_add_into`, `scaled_assign_into`, `tensoradd_add_into`,
  `tensoradd_assign_into`, `tensoradd_execute_with`, `tensoradd_fusion_into`,
  `tensoradd_fusion_into_with`, `tensoradd_fusion_into_with_context`,
  `tensoradd_into_with`, `tensoradd_into_with_backend_and_conjugation`,
  `tensoradd_into_with_conjugation`, `tensorcopy_into`, `tensorcopy_into_with`,
  `tensortrace_execute_with`, `tensortrace_fusion_execute_with`,
  `tensortrace_fusion_into`, `tensortrace_fusion_into_with`,
  `tensortrace_into_with`, `tree_transform_execute_with`,
  `tree_transform_into`, `tree_transform_into_with`,
  `tree_transform_into_with_context`, `tree_transform_overwrite_execute_with`,
  `tree_transform_overwrite_into`, `tree_transform_overwrite_into_with`,
  `tree_transform_overwrite_into_with_context`, `tree_transform_structure`.
- `ReportsPlacement`, `TreeTransformReplayProfile`, `TreeTransformStructure`,
  `tensoradd_structure`, `tensoradd_structure_with_conjugation`,
  `TensorAddStructure`, `TensorAddStructureTerm`, `ConjugateValue`,
  `DenseBlockScalar`, `DenseRecouplingScalar`, `RealStructuralCoefficient`,
  `RecouplingCoefficientAction`, `TreeTransformScalar`,
  `DenseTreeTransformOperations`, `HostAllocator`, `HostTensorOperations`,
  `HostTensorOperationsBackend`, `HostTensorOperationsWorkspace`,
  `HostTreeTransformBackend`, `TensorOperationsBackend`,
  `TreeTransformBackend`, `HostTreeTransformWorkspace`,
  `TreeTransformWorkspace`.
- `tensortrace_fusion_dyn_into`, `tensortrace_fusion_dyn_into_checked`,
  `tensortrace_fusion_dyn_owned`, `tensortrace_fusion_dyn_owned_checked`,
  `tensortrace_fusion_dyn_selected_homspace_checked`,
  `tensortrace_fusion_structure`, `tensortrace_structure`,
  `TensorTraceFusionStructure`, `TensorTraceFusionStructureTerm`,
  `TensorTraceStructure`, `TensorTraceStructureTerm`.
- `build_all_codomain_tree_transform_group_plan`,
  `build_generic_tree_pair_transform_group_plan`,
  `build_tree_pair_transform_group_plan`, `build_tree_transform_group_plan`,
  `RuntimeTreeTransformCacheInfo`, `RuntimeTreeTransformStore`,
  `TreePairTransformCache`, `TreeTransformBlockSpec`, `TreeTransformCache`,
  `TreeTransformCacheStats`, `TreeTransformGroupBlockSpec`,
  `TreeTransformGroupPlan`, `TreeTransformKeyBlockSpec`,
  `TreeTransformOperationKind`, `TreeTransformRuleCacheKey`.
- Conditional/hidden exports: `cuda` (`cfg(feature = "cuda")`),
  `try_cat_owned_c64_raw`, `try_cat_owned_raw`, `OwnedCatC64Source`,
  `OwnedCatCopy`, `OwnedCatSide` (`doc(hidden)`).

### `tenet::matrixalgebra` → `tenet_matrixalgebra`

- `diagonal_bond_bound_space`, `diagonal_bond_bound_space_generic`,
  `diagonal_bond_bound_space_like`, `diagonal_bond_data`, `eig_full`,
  `eig_full_dyn`, `eig_trunc`, `eig_trunc_dyn`, `eig_vals`, `eig_vals_dyn`,
  `eigh_full`, `eigh_full_dyn`, `eigh_trunc`, `eigh_trunc_dyn`, `eigh_vals`,
  `eigh_vals_dyn`, `left_null`, `left_null_dyn`, `left_orth`, `left_polar`,
  `left_polar_dyn`, `lq_compact`, `lq_compact_dyn`,
  `lq_compact_dyn_generic`, `lq_full`, `lq_full_dyn`, `qr_compact`,
  `qr_compact_dyn`, `qr_compact_dyn_generic`, `qr_full`, `qr_full_dyn`,
  `right_null`, `right_null_dyn`, `right_orth`, `right_polar`,
  `right_polar_dyn`, `scale_axis_by_spectrum`,
  `scale_axis_by_spectrum_mapped`, `svd_compact_dyn`,
  `svd_compact_factors_dyn`, `svd_compact_factors_dyn_generic`, `svd_full`,
  `svd_full_dyn`, `svd_trunc`, `svd_trunc_dyn`, `svd_trunc_dyn_generic`,
  `svd_trunc_factors_dyn`, `svd_trunc_factors_dyn_generic`, `svd_vals`,
  `svd_vals_dyn`, `svd_vals_dyn_generic`, `BoundDynFactor`,
  `BoundDynamicTensorRef`, `EigFull`, `EigFullDyn`, `EigTrunc`,
  `EigTruncDyn`, `EighFull`, `EighFullDyn`, `EighTrunc`, `EighTruncDyn`,
  `FactorScalar`, `SpectrumMagnitude`, `SvdCompactDyn`, `SvdFactorsDyn`,
  `SvdFull`, `SvdFullDyn`, `SvdTrunc`, `SvdTruncDyn`, `SvdTruncFactorsDyn`,
  `exp`, `exp_dyn`, `inv`, `inv_direct_dyn`, `inv_dyn`, `pinv`, `pinv_dyn`,
  `select_truncation`, `Truncation`, `TruncationDecision`,
  `TruncationError`, `TruncationSpace`, `WeightedSpectrum`, and the
  `truncation` module.
- Hidden/conditional exports: `validate_hermitian_regions` (`doc(hidden)`),
  `sector_matricization_diagnostic` and `SectorMatricizationDiagnostic`
  (`cfg(feature = "diagnostics")`, `doc(hidden)`).

The direct crates are broader and unstable. No compatibility wrappers are
provided; callers should migrate imports before the first stable release.
