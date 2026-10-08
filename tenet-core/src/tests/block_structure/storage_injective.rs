use super::*;

fn unbound_expert_space<const NOUT: usize, const NIN: usize>(
    blocks: Vec<BlockSpec>,
) -> Result<FusionTensorMapSpace<NOUT, NIN>, CoreError> {
    // Storage-only fixtures use opaque ordinals deliberately: categorical
    // key and shape admission is covered independently by try_bind_rule.
    let dense = TensorMapSpace::<NOUT, NIN>::from_dims([1; NOUT], [1; NIN]).unwrap();
    let homspace =
        FusionTreeHomSpace::from_sector_ids((0..NOUT).map(|_| (0, 1)), (0..NIN).map(|_| (0, 1)));
    FusionTensorMapSpace::new_unbound(
        dense,
        homspace,
        BlockStructure::from_blocks_with_rank(NOUT + NIN, blocks).unwrap(),
    )
}

#[test]
fn expert_fusion_space_rejects_self_overlapping_storage() {
    // What: an owning symmetric space cannot assign two logical elements of
    // one block to the same physical element.
    for (block, offset) in [
        (
            BlockSpec::with_key(BlockKey::ordinal(0), vec![2, 2], vec![1, 1], 0).unwrap(),
            1,
        ),
        (
            BlockSpec::with_key(BlockKey::ordinal(0), vec![2, 1], vec![0, 1], 0).unwrap(),
            0,
        ),
    ] {
        assert_eq!(
            unbound_expert_space::<1, 1>(vec![block]),
            Err(CoreError::OverlappingBlockStorage {
                first_block: 0,
                second_block: 0,
                offset,
            })
        );
    }
}

#[test]
fn expert_fusion_space_rejects_cross_block_storage_aliases() {
    // What: distinct logical symmetric blocks cannot own the same physical
    // destination element, and diagnostics preserve caller block order.
    let blocks = vec![
        BlockSpec::with_key(BlockKey::ordinal(0), vec![2], vec![2], 0).unwrap(),
        BlockSpec::with_key(BlockKey::ordinal(1), vec![2], vec![1], 1).unwrap(),
    ];
    assert_eq!(
        unbound_expert_space::<1, 0>(blocks),
        Err(CoreError::OverlappingBlockStorage {
            first_block: 0,
            second_block: 1,
            offset: 2,
        })
    );
}

#[test]
fn expert_fusion_space_accepts_exact_non_overlapping_strided_storage() {
    // What: expert admission retains arbitrary legal layouts, including
    // layouts that a conservative sorted-span proof cannot establish.
    let rank_two_cases = [
        vec![BlockSpec::with_key(BlockKey::ordinal(0), vec![3, 2], vec![2, 3], 0).unwrap()],
        vec![BlockSpec::with_key(BlockKey::ordinal(0), vec![2, 3], vec![3, 1], 0).unwrap()],
        vec![BlockSpec::with_key(BlockKey::ordinal(0), vec![1, 2], vec![0, 1], 0).unwrap()],
    ];
    reset_exact_storage_fallback_count();
    for blocks in rank_two_cases {
        unbound_expert_space::<1, 1>(blocks).unwrap();
    }
    assert!(exact_storage_fallback_count() > 0);

    let rank_one_cases = [
        vec![
            BlockSpec::with_key(BlockKey::ordinal(0), vec![4], vec![2], 0).unwrap(),
            BlockSpec::with_key(BlockKey::ordinal(1), vec![4], vec![2], 1).unwrap(),
        ],
        vec![
            BlockSpec::with_key(BlockKey::ordinal(0), vec![2], vec![1], 4).unwrap(),
            BlockSpec::with_key(BlockKey::ordinal(1), vec![2], vec![1], 0).unwrap(),
        ],
        vec![
            BlockSpec::with_key(BlockKey::ordinal(0), vec![0], vec![0], 0).unwrap(),
            BlockSpec::with_key(BlockKey::ordinal(1), vec![1], vec![1], 0).unwrap(),
        ],
    ];
    for blocks in rank_one_cases {
        unbound_expert_space::<1, 0>(blocks).unwrap();
    }

    unbound_expert_space::<0, 0>(vec![BlockSpec::with_key(
        BlockKey::ordinal(0),
        vec![],
        vec![],
        0,
    )
    .unwrap()])
    .unwrap();
    unbound_expert_space::<1, 0>(vec![BlockSpec::with_key(
        BlockKey::ordinal(0),
        vec![0],
        vec![1],
        0,
    )
    .unwrap()])
    .unwrap();
}

#[test]
fn general_block_structure_retains_aliasing_read_view_contract() {
    // What: storage ownership admission belongs to FusionTensorMapSpace;
    // general block metadata remains usable for intentionally aliased views.
    let structure = BlockStructure::from_blocks(vec![
        BlockSpec::with_key(BlockKey::ordinal(0), vec![2], vec![1], 0).unwrap(),
        BlockSpec::with_key(BlockKey::ordinal(1), vec![2], vec![1], 0).unwrap(),
    ])
    .unwrap();
    assert_eq!(structure.required_len().unwrap(), 2);
}

fn assert_expert_storage_admission_for_rule<R>(rule: &R, sector: SectorId)
where
    R: MultiplicityFreeFusionRule,
{
    let homspace = FusionTreeHomSpace::from_sectors([(sector, 2)], [(sector, 3)]);
    let key = homspace.fusion_tree_keys(rule)[0].clone();
    let structure = BlockStructure::from_blocks(vec![BlockSpec::with_key(
        BlockKey::FusionTree(key),
        vec![2, 3],
        vec![1, 4],
        2,
    )
    .unwrap()])
    .unwrap();
    FusionTensorMapSpace::new_unbound(
        TensorMapSpace::<1, 1>::from_dims([2], [3]).unwrap(),
        homspace,
        structure,
    )
    .unwrap()
    .try_bind_rule(rule)
    .unwrap();
}

#[test]
fn expert_storage_admission_is_symmetry_independent() {
    // What: the same exact custom-layout admission and categorical binding
    // succeeds for U(1), SU(2), fZ2, and their nested product.
    assert_expert_storage_admission_for_rule(&U1FusionRule, u1(0));
    assert_expert_storage_admission_for_rule(&SU2FusionRule, su2(0));
    assert_expert_storage_admission_for_rule(&FermionParityFusionRule, z2_even());

    type Fz2U1 = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
    type Triple = ProductFusionRule<Fz2U1, SU2FusionRule>;
    let pair = Fz2U1::new(FermionParityFusionRule, U1FusionRule);
    let pair_vacuum = pair.encode_component_ids(z2_even(), u1(0));
    let triple = Triple::new(pair, SU2FusionRule);
    let vacuum = triple.encode_component_ids(pair_vacuum, su2(0));
    assert_expert_storage_admission_for_rule(&triple, vacuum);
}

#[cfg(feature = "racah-generated")]
#[test]
fn checked_generic_coupled_storage_is_admitted_by_witness() {
    // What: checked-Generic SU(3) destinations with outer multiplicity use
    // the same coupled-sector builder, so both the staged preview and the
    // committed structure carry the witness.
    let rule = SUNFusionRule::new(3).unwrap();
    let fundamental = rule.encode_dynkin(&[1, 0]).unwrap();
    let adjoint = rule.encode_dynkin(&[1, 1]).unwrap();
    let trivial = rule.encode_dynkin(&[0, 0]).unwrap();
    let leg = |dual| SectorLeg::new([(fundamental, 3), (adjoint, 2), (trivial, 2)], dual);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(false), leg(true)]),
        FusionProductSpace::new([leg(false), leg(true)]),
    );
    let prepared = hom
        .prepare_coupled_subblock_structure_from_leg_degeneracies_generic_checked(&rule)
        .unwrap();
    assert_canonical_storage_admitted_without_enumeration(prepared.structure());
    assert_canonical_storage_admitted_without_enumeration(&prepared.commit());
}

#[test]
fn external_aliased_storage_keeps_exact_rejection() {
    // What: layouts without the construction witness keep the exact check
    // and its error, even when they interleave like coupled subblocks.
    let aliased = BlockStructure::from_blocks(vec![
        BlockSpec::with_key(BlockKey::ordinal(0), vec![2, 2], vec![1, 3], 0).unwrap(),
        BlockSpec::with_key(BlockKey::ordinal(1), vec![2, 2], vec![1, 3], 1).unwrap(),
    ])
    .unwrap();
    assert!(!aliased.storage_tiling_proven());
    reset_exact_storage_fallback_count();
    assert_eq!(
        validate_block_storage_injective(&aliased),
        Err(CoreError::OverlappingBlockStorage {
            first_block: 0,
            second_block: 1,
            offset: 1,
        })
    );
    assert!(exact_storage_fallback_count() > 0);
}
