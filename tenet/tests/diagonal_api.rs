use std::sync::Arc;

use tenet::core::{SU2FusionRule, SU2Irrep};
use tenet::prelude::{Complex64, Dtype, Error, Runtime, Scalar, Space, Tensor};
use tenet::typed::{GradedSpace, SectorSpectrum, TensorMap};

fn runtime() -> Runtime {
    Runtime::builder().dense_threads(1).build().unwrap()
}

#[test]
fn erased_diagonal_preserves_canonical_positions_and_dual_leg() {
    let runtime = runtime();
    let bond = Space::u1([(-1, 2), (1, 1)]).dual();
    let values: Vec<Vec<Scalar>> = bond
        .sectors()
        .iter()
        .enumerate()
        .map(|(sector, &(_, degeneracy))| {
            (0..degeneracy)
                .map(|index| {
                    Scalar::C64(Complex64::new((10 * sector + index) as f64, index as f64))
                })
                .collect()
        })
        .collect();
    let tensor = Tensor::diagonal(&runtime, Dtype::C64, &bond, values.clone()).unwrap();
    assert_eq!(tensor.codomain_spaces()[0], bond);
    assert_eq!(tensor.domain_spaces()[0], bond);
    assert!(tensor.is_diagonal(0.0).unwrap());
    assert_eq!(tensor.diagonal_spectrum().unwrap().unwrap(), values);
    let mut wrong_dtype = values.clone();
    wrong_dtype[0][0] = Scalar::F64(1.0);
    assert!(matches!(
        Tensor::diagonal(&runtime, Dtype::C64, &bond, wrong_dtype),
        Err(Error::DtypeMismatch)
    ));

    let product = Space::product([((0, 0), 1), ((1, 1), 2)]).unwrap();
    let product_values: Vec<Vec<Scalar>> = product
        .sectors()
        .iter()
        .enumerate()
        .map(|(sector, &(_, degeneracy))| {
            (0..degeneracy)
                .map(|index| Scalar::F64((10 * sector + index) as f64))
                .collect()
        })
        .collect();
    assert_eq!(
        Tensor::diagonal(&runtime, Dtype::F64, &product, product_values.clone())
            .unwrap()
            .diagonal_spectrum()
            .unwrap()
            .unwrap(),
        product_values
    );
}

#[test]
fn real_c64_readback_stays_compact() {
    let runtime = runtime();
    let v = Space::u1([(0, 2)]);
    let source = Tensor::from_block_fn(&runtime, [&v], [&v], |_, index| {
        Complex64::new(
            if index[0] == index[1] {
                (index[0] + 1) as f64
            } else {
                0.0
            },
            0.0,
        )
    })
    .unwrap();
    let (diagonal, _) = source.eigh_full().unwrap();
    assert_eq!(
        diagonal.diagonal_spectrum().unwrap().unwrap()[0],
        vec![
            Scalar::C64(Complex64::new(2.0, 0.0)),
            Scalar::C64(Complex64::new(1.0, 0.0))
        ]
    );
}

#[test]
fn typed_diagonal_canonicalizes_labels_and_dense_predicate_handles_nonfinite_offdiag() {
    let runtime = runtime();
    let rule = Arc::new(SU2FusionRule);
    let bond = GradedSpace::try_new(
        Arc::clone(&rule),
        [
            (SU2Irrep::from_twice_spin(2), 1),
            (SU2Irrep::from_twice_spin(0), 2),
        ],
        false,
    )
    .unwrap();
    let tensor = TensorMap::<SU2FusionRule, f64>::diagonal(
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: SU2Irrep::from_twice_spin(2),
                values: vec![3.0],
            },
            SectorSpectrum {
                sector: SU2Irrep::from_twice_spin(0),
                values: vec![1.0, 2.0],
            },
        ],
    )
    .unwrap();
    assert_eq!(
        tensor.diagonal_spectrum().unwrap().unwrap()[0].sector,
        SU2Irrep::from_twice_spin(0)
    );
    assert!(tensor.is_diagonal(0.0).unwrap());
    assert!(TensorMap::<SU2FusionRule, f64>::diagonal(
        &runtime,
        &bond,
        [SectorSpectrum {
            sector: SU2Irrep::from_twice_spin(0),
            values: vec![1.0, 2.0],
        }],
    )
    .is_err());
    assert!(TensorMap::<SU2FusionRule, f64>::diagonal(
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: SU2Irrep::from_twice_spin(0),
                values: vec![1.0],
            },
            SectorSpectrum {
                sector: SU2Irrep::from_twice_spin(2),
                values: vec![3.0],
            },
        ],
    )
    .is_err());
    assert!(TensorMap::<SU2FusionRule, f64>::diagonal(
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: SU2Irrep::from_twice_spin(0),
                values: vec![1.0, 2.0],
            },
            SectorSpectrum {
                sector: SU2Irrep::from_twice_spin(4),
                values: vec![3.0],
            },
        ],
    )
    .is_err());
    assert!(TensorMap::<SU2FusionRule, f64>::diagonal(
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: SU2Irrep::from_twice_spin(0),
                values: vec![1.0, 2.0],
            },
            SectorSpectrum {
                sector: SU2Irrep::from_twice_spin(0),
                values: vec![3.0, 4.0],
            },
        ],
    )
    .is_err());

    let v = Space::u1([(0, 2)]);
    let dense = Tensor::from_block_fn(&runtime, [&v], [&v], |_, index| {
        if index == [0, 1] {
            f64::NAN
        } else {
            0.0
        }
    })
    .unwrap();
    assert!(!dense.is_diagonal(1.0).unwrap());
    let finite = Tensor::from_block_fn(&runtime, [&v], [&v], |_, index| {
        if index[0] == index[1] {
            4.0
        } else if index == [0, 1] {
            0.25
        } else {
            0.0
        }
    })
    .unwrap();
    assert!(!finite.is_diagonal(0.05).unwrap());
    assert!(finite.is_diagonal(0.1).unwrap());
    let exact = Tensor::id(&runtime, Dtype::F64, [&v]).unwrap();
    assert!(exact.is_diagonal(0.0).unwrap());
    let inf = Tensor::from_block_fn(&runtime, [&v], [&v], |_, index| {
        if index == [1, 0] {
            f64::INFINITY
        } else {
            0.0
        }
    })
    .unwrap();
    assert!(!inf.is_diagonal(1.0).unwrap());
    assert!(dense.is_diagonal(-1.0).is_err());
    let vector = Tensor::zeros(&runtime, Dtype::F64, [&v], []).unwrap();
    assert!(vector.is_diagonal(f64::NAN).is_err());
}
