use tenet::prelude::{
    Complex64, GradedSpace, PhysicalDense, Runtime, SU2FusionRule, SU2Irrep, TensorMap,
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
    assert!(target.project_physical_dense(&bad_shape).is_err());
    assert_eq!(target.data(), before);

    let bad_length = PhysicalDense {
        shape: vec![2, 2, 2, 2],
        data: vec![0.0; 15],
    };
    assert!(target.project_physical_dense(&bad_length).is_err());
    assert_eq!(target.data(), before);
}
