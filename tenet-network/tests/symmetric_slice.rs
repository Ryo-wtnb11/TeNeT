use tenet::core::{FermionParityFusionRule, SectorId, SectorLeg, U1FusionRule, U1Irrep, Z2Irrep};
use tenet_network::{
    DegeneracyRange, NetworkIR, SectorSlice, SliceError, SliceKind, SymmetricSlicePlan,
    SymmetricSliceSpec, TemporaryLabel, TensorAxis, TensorId,
};

fn label(name: &str) -> TemporaryLabel {
    TemporaryLabel::new(name)
}

fn range(start: usize, end: usize) -> DegeneracyRange {
    DegeneracyRange::new(start, end).unwrap()
}

fn piece(sector: usize, start: usize, end: usize) -> SectorSlice {
    SectorSlice::new(SectorId::new(sector), range(start, end))
}

fn axis(tensor: usize) -> TensorAxis {
    TensorAxis::new(TensorId::new(tensor), 0)
}

fn internal_ir() -> NetworkIR {
    NetworkIR::from_labels(vec![vec![label("x")], vec![label("x")]], vec![]).unwrap()
}

fn spec(
    label_name: &str,
    authority: TensorAxis,
    leg: SectorLeg,
    pieces: Vec<SectorSlice>,
) -> SymmetricSliceSpec {
    SymmetricSliceSpec::new(label(label_name), authority, leg, pieces)
}

#[test]
fn canonicalizes_input_and_piece_order_without_merging_adjacent_ranges() {
    let ir = NetworkIR::from_labels(
        vec![vec![label("x")], vec![label("x")], vec![label("y")]],
        vec![label("y")],
    )
    .unwrap();
    let x_leg = SectorLeg::new([(SectorId::new(1), 2), (SectorId::new(3), 1)], false);
    let y_leg = SectorLeg::new([(SectorId::new(2), 2)], false);

    let x_forward = vec![piece(1, 0, 1), piece(1, 1, 2), piece(3, 0, 1)];
    let x_shuffled = vec![piece(3, 0, 1), piece(1, 1, 2), piece(1, 0, 1)];
    let first = SymmetricSlicePlan::try_new(
        &ir,
        vec![
            spec("y", axis(2), y_leg.clone(), vec![piece(2, 0, 2)]),
            spec("x", axis(0), x_leg.clone(), x_shuffled),
        ],
    )
    .unwrap();
    let second = SymmetricSlicePlan::try_new(
        &ir,
        vec![
            spec("x", axis(0), x_leg, x_forward.clone()),
            spec("y", axis(2), y_leg, vec![piece(2, 0, 2)]),
        ],
    )
    .unwrap();

    assert_eq!(first, second);
    assert_eq!(first.indices()[0].label(), &label("x"));
    assert_eq!(first.indices()[0].kind(), SliceKind::Internal);
    assert_eq!(first.indices()[1].kind(), SliceKind::Output);
    assert_eq!(first.indices()[0].output_position(), None);
    assert_eq!(first.indices()[1].output_position(), Some(0));
    assert_eq!(first.indices()[0].pieces(), x_forward);
    assert_eq!(first.indices()[0].pieces().len(), 3);

    let merged = SymmetricSlicePlan::try_new(
        &ir,
        vec![
            spec(
                "x",
                axis(0),
                SectorLeg::new([(SectorId::new(1), 2), (SectorId::new(3), 1)], false),
                vec![piece(1, 0, 2), piece(3, 0, 1)],
            ),
            spec(
                "y",
                axis(2),
                SectorLeg::new([(SectorId::new(2), 2)], false),
                vec![piece(2, 0, 2)],
            ),
        ],
    )
    .unwrap();
    assert_ne!(first, merged, "adjacent partition boundaries are identity");
}

#[test]
fn counts_and_enumerates_multi_sector_multi_index_cartesian_product() {
    let ir = NetworkIR::from_labels(
        vec![
            vec![label("x")],
            vec![label("x")],
            vec![label("y")],
            vec![label("y")],
        ],
        vec![],
    )
    .unwrap();
    let plan = SymmetricSlicePlan::try_new(
        &ir,
        vec![
            spec(
                "y",
                axis(2),
                SectorLeg::new([(SectorId::new(4), 2)], false),
                vec![piece(4, 0, 1), piece(4, 1, 2)],
            ),
            spec(
                "x",
                axis(0),
                SectorLeg::new([(SectorId::new(1), 1), (SectorId::new(3), 1)], false),
                vec![piece(3, 0, 1), piece(1, 0, 1)],
            ),
        ],
    )
    .unwrap();

    assert_eq!(plan.nslices(), 4);
    let combinations = plan
        .combinations()
        .map(|items| {
            items
                .into_iter()
                .map(|item| (item.sector().id(), item.range().start()))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        combinations,
        vec![
            vec![(1, 0), (4, 0)],
            vec![(1, 0), (4, 1)],
            vec![(3, 0), (4, 0)],
            vec![(3, 0), (4, 1)],
        ]
    );
}

#[test]
fn rejects_every_range_partition_and_authority_error() {
    assert_eq!(
        DegeneracyRange::new(2, 2),
        Err(SliceError::EmptyRange { at: 2 })
    );
    assert_eq!(
        DegeneracyRange::new(3, 2),
        Err(SliceError::ReversedRange { start: 3, end: 2 })
    );

    let ir = internal_ir();
    let leg = SectorLeg::new([(SectorId::new(1), 3)], false);
    let make = |pieces| spec("x", axis(0), leg.clone(), pieces);
    assert!(matches!(
        SymmetricSlicePlan::try_new(
            &ir,
            vec![spec("missing", axis(0), leg.clone(), vec![piece(1, 0, 3)])]
        ),
        Err(SliceError::UnknownLabel(_))
    ));
    assert!(matches!(
        SymmetricSlicePlan::try_new(
            &ir,
            vec![make(vec![piece(1, 0, 3)]), make(vec![piece(1, 0, 3)])]
        ),
        Err(SliceError::DuplicateLabel(_))
    ));
    assert!(matches!(
        SymmetricSlicePlan::try_new(
            &ir,
            vec![spec("x", axis(1), leg.clone(), vec![piece(1, 0, 3)])]
        ),
        Err(SliceError::InvalidAuthority { .. })
    ));
    assert!(matches!(
        SymmetricSlicePlan::try_new(&ir, vec![make(vec![piece(9, 0, 1)])]),
        Err(SliceError::UnknownSector { .. })
    ));
    assert!(matches!(
        SymmetricSlicePlan::try_new(&ir, vec![make(vec![piece(1, 0, 4)])]),
        Err(SliceError::RangeOutOfBounds { .. })
    ));
    assert!(matches!(
        SymmetricSlicePlan::try_new(&ir, vec![make(vec![piece(1, 0, 2), piece(1, 1, 3)])]),
        Err(SliceError::OverlappingRanges { .. })
    ));
    assert!(matches!(
        SymmetricSlicePlan::try_new(&ir, vec![make(vec![piece(1, 0, 1), piece(1, 2, 3)])]),
        Err(SliceError::IncompleteCoverage { .. })
    ));
}

#[test]
fn distinguishes_empty_leg_from_no_sliced_label() {
    let ir = internal_ir();
    let unsliced = SymmetricSlicePlan::try_new(&ir, vec![]).unwrap();
    let empty_leg = SymmetricSlicePlan::try_new(
        &ir,
        vec![spec(
            "x",
            axis(0),
            SectorLeg::new(Vec::<(SectorId, usize)>::new(), false),
            vec![],
        )],
    )
    .unwrap();

    assert!(unsliced.is_empty());
    assert_eq!(unsliced.nslices(), 1);
    assert_eq!(unsliced.combinations().count(), 1);
    assert!(!empty_leg.is_empty());
    assert_eq!(empty_leg.nslices(), 0);
    assert_eq!(empty_leg.combinations().count(), 0);
}

#[test]
fn canonical_effective_occurrence_owns_non_self_dual_sector_identity() {
    let ir = internal_ir();
    let charge = U1Irrep::new(1);
    let authority_leg = SectorLeg::new([(charge, 2)], false);
    let partner_leg = authority_leg.dual(&U1FusionRule);
    let plan = SymmetricSlicePlan::try_new(
        &ir,
        vec![spec(
            "x",
            axis(0),
            authority_leg.clone(),
            vec![SectorSlice::new(charge.into(), range(0, 2))],
        )],
    )
    .unwrap();

    // `axis(0)` means the first effective occurrence after adjoint rotation;
    // the partner uses the validated dual space, not another raw schema id.
    assert_eq!(plan.indices()[0].authority(), axis(0));
    assert_eq!(plan.indices()[0].authority_leg(), &authority_leg);
    assert_eq!(partner_leg.sectors(), &[SectorId::from(U1Irrep::new(-1))]);
    assert!(partner_leg.is_dual());
}

#[test]
fn fermion_descriptor_is_coefficient_free_and_structural_zeros_are_valid() {
    let _fermion_rule = FermionParityFusionRule;
    let odd: SectorId = Z2Irrep::ODD.into();
    let ir = NetworkIR::from_labels(
        vec![
            vec![label("f")],
            vec![label("f")],
            vec![label("z")],
            vec![label("z")],
        ],
        vec![],
    )
    .unwrap();
    let plan = SymmetricSlicePlan::try_new(
        &ir,
        vec![
            SymmetricSliceSpec::new(
                label("f"),
                axis(0),
                SectorLeg::new([(odd, 1)], false),
                vec![SectorSlice::new(odd, range(0, 1))],
            ),
            spec(
                "z",
                axis(2),
                SectorLeg::new([(SectorId::new(8), 2)], false),
                vec![piece(8, 0, 1), piece(8, 1, 2)],
            ),
        ],
    )
    .unwrap();

    // The schema accepts all Cartesian combinations, including combinations
    // with no admissible tensor block. Fusion trees and fermionic signs are not
    // represented or modified here.
    assert_eq!(plan.nslices(), 2);
    assert_eq!(plan.indices()[0].pieces()[0].sector(), odd);
}
