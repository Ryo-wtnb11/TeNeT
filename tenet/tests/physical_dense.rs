use tenet::operations::OperationError;
use tenet::prelude::{
    Complex64, GradedSpace, PhysicalDense, PhysicalDenseError, Runtime, SU2FusionRule, SU2Irrep,
    TensorMap, U1FusionRule, U1Irrep,
};

fn su2_multitree<D: tenet::prelude::TensorScalar>(
    runtime: &Runtime,
    value: impl Fn(&[SU2Irrep]) -> D,
) -> TensorMap<SU2FusionRule, D> {
    let half = SU2Irrep::from_twice_spin(1);
    let leg = GradedSpace::try_new(SU2FusionRule, [(half, 1)]).unwrap();
    TensorMap::from_block_fn(runtime, [&leg, &leg, &leg], [&leg], |trees, _| {
        value(trees.codomain_innerlines())
    })
    .unwrap()
}

fn assert_complex_close(actual: &[Complex64], expected: &[Complex64]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!((actual - expected).norm() < 2.0e-12);
    }
}

fn assert_real_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() < 2.0e-12);
    }
}

#[test]
fn physical_dense_roundtrips_real_and_complex_su2_multitree_data() {
    let runtime = Runtime::builder().build().unwrap();
    let real = su2_multitree(&runtime, |inner| {
        if inner[0].twice_spin() == 0 {
            1.25
        } else {
            -0.75
        }
    });
    let real_reduced = real.data().to_vec();
    let physical = real.to_physical_dense().unwrap();
    assert_eq!(physical.shape, [2, 2, 2, 2]);
    assert_eq!(real.data(), real_reduced);
    assert_real_close(
        real.project_physical_dense(&physical).unwrap().data(),
        &real_reduced,
    );

    let complex = su2_multitree(&runtime, |inner| {
        if inner[0].twice_spin() == 0 {
            Complex64::new(1.25, -0.5)
        } else {
            Complex64::new(-0.75, 0.25)
        }
    });
    let complex_reduced = complex.data().to_vec();
    let physical = complex.to_physical_dense().unwrap();
    assert_eq!(physical.shape, [2, 2, 2, 2]);
    let projected = complex.project_physical_dense(&physical).unwrap();
    assert_complex_close(projected.data(), &complex_reduced);
    assert_eq!(complex.data(), complex_reduced);
}

#[test]
fn projection_validates_shape_and_length_without_changing_target() {
    let runtime = Runtime::builder().build().unwrap();
    let target = su2_multitree(&runtime, |_| 9.0);
    let before = target.data().to_vec();

    let bad_shape = PhysicalDense {
        shape: vec![2, 2],
        data: vec![0.0; 4],
    };
    let error = target.project_physical_dense(&bad_shape).unwrap_err();
    assert!(matches!(
        error,
        PhysicalDenseError::Operation(OperationError::ShapeMismatch { .. })
    ));
    assert_eq!(target.data(), before);

    let bad_length = PhysicalDense {
        shape: vec![2, 2, 2, 2],
        data: vec![0.0; 15],
    };
    let error = target.project_physical_dense(&bad_length).unwrap_err();
    assert!(matches!(
        error,
        PhysicalDenseError::Operation(OperationError::ElementCountMismatch { .. })
    ));
    assert_eq!(target.data(), before);
}

#[test]
fn physical_entries_match_the_executable_tensorkit_su2_oracle() {
    let runtime = Runtime::builder().build().unwrap();
    let real = su2_multitree(&runtime, |inner| f64::from(inner[0].twice_spin() == 2));
    let inv_sqrt_6 = 1.0 / 6.0_f64.sqrt();
    let expected_real = [
        0.0,
        -inv_sqrt_6,
        -inv_sqrt_6,
        0.0,
        2.0 * inv_sqrt_6,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -2.0 * inv_sqrt_6,
        0.0,
        inv_sqrt_6,
        inv_sqrt_6,
        0.0,
    ];
    assert_real_close(&real.to_physical_dense().unwrap().data, &expected_real);

    let scale = Complex64::new(1.0, 2.0);
    let complex = su2_multitree(&runtime, |inner| {
        if inner[0].twice_spin() == 0 {
            scale
        } else {
            Complex64::new(0.0, 0.0)
        }
    });
    let inv_sqrt_2 = 1.0 / 2.0_f64.sqrt();
    let mut expected_complex = vec![Complex64::new(0.0, 0.0); 16];
    for (index, coefficient) in [
        (1, -inv_sqrt_2),
        (2, inv_sqrt_2),
        (13, -inv_sqrt_2),
        (14, inv_sqrt_2),
    ] {
        expected_complex[index] = scale * coefficient;
    }
    assert_complex_close(
        &complex.to_physical_dense().unwrap().data,
        &expected_complex,
    );
}

fn permute_each_axis<D: Copy>(physical: &PhysicalDense<D>, target_to_source: &[usize]) -> Vec<D> {
    assert_eq!(physical.shape, [target_to_source.len(); 2]);
    let dimension = target_to_source.len();
    (0..dimension * dimension)
        .map(|target| {
            let i = target % dimension;
            let j = target / dimension;
            physical.data[target_to_source[i] + dimension * target_to_source[j]]
        })
        .collect()
}

#[test]
fn su2_spin_one_singlet_projects_to_explicit_doubled_u1_charges() {
    let runtime = Runtime::builder().build().unwrap();
    let spin_one = SU2Irrep::from_twice_spin(2);
    let su2_leg = GradedSpace::try_new(SU2FusionRule, [(spin_one, 1)]).unwrap();
    let singlet: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&su2_leg, &su2_leg], [], |_, _| 1.0).unwrap();
    let su2_physical = singlet.to_physical_dense().unwrap();
    assert_eq!(su2_physical.shape, [3, 3]);

    // SU(2) uses doubled magnetic labels (2, 0, -2), whereas this U(1)
    // leg's canonical order is (0, -2, 2). Keep the basis change explicit.
    let target_to_source = [1, 2, 0];
    let u1_physical = PhysicalDense {
        shape: vec![3, 3],
        data: permute_each_axis(&su2_physical, &target_to_source),
    };
    let u1_leg = GradedSpace::try_new(
        U1FusionRule,
        [
            (U1Irrep::new(0), 1),
            (U1Irrep::new(-2), 1),
            (U1Irrep::new(2), 1),
        ],
    )
    .unwrap();
    assert_eq!(
        u1_leg.sectors().unwrap(),
        [U1Irrep::new(0), U1Irrep::new(-2), U1Irrep::new(2)]
    );
    let target: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&u1_leg, &u1_leg], []).unwrap();
    let projected = target.project_physical_dense(&u1_physical).unwrap();

    let inv_sqrt_3 = 1.0 / 3.0_f64.sqrt();
    let mut labelled = projected
        .blocks()
        .unwrap()
        .map(|(trees, block)| {
            (
                (
                    trees.codomain_uncoupled()[0].charge(),
                    trees.codomain_uncoupled()[1].charge(),
                ),
                *block.get(&[0, 0]).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    labelled.sort_unstable_by_key(|entry| entry.0);
    let expected = [
        ((-2, 2), inv_sqrt_3),
        ((0, 0), -inv_sqrt_3),
        ((2, -2), inv_sqrt_3),
    ];
    assert_eq!(labelled.len(), expected.len());
    for ((labels, actual), (expected_labels, expected_value)) in labelled.into_iter().zip(expected)
    {
        assert_eq!(labels, expected_labels);
        assert!((actual - expected_value).abs() < 2.0e-12);
    }
    assert_real_close(
        &projected.to_physical_dense().unwrap().data,
        &u1_physical.data,
    );
}
