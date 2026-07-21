use std::collections::HashMap;
use tenet::prelude::*;

fn u1_value(key: &BlockKey, indices: &[usize], bias: f64) -> f64 {
    let BlockKey::FusionTree(key) = key else {
        unreachable!("user tensors use fusion-tree keys")
    };
    u1_tree_value(key, indices, bias)
}

fn u1_tree_value(key: &FusionTreePairKey, indices: &[usize], bias: f64) -> f64 {
    bias + key.codomain_uncoupled()[0].id() as f64 * 100.0
        + indices
            .iter()
            .enumerate()
            .map(|(axis, &index)| index as f64 * 10f64.powi(axis as i32))
            .sum::<f64>()
}

#[test]
fn absorb_copies_exact_shared_block_prefixes_and_preserves_destination_remainder() {
    // What: shared U1 keys receive their per-axis minimum prefix while
    // destination-only keys and coordinates keep their original bits.
    let runtime = Runtime::builder().build().unwrap();
    let destination_space = Space::u1([(-1, 1), (0, 3), (1, 2)]);
    let source_space = Space::u1([(0, 2), (1, 4), (2, 1)]);
    let destination = Tensor::from_block_fn(
        &runtime,
        [&destination_space],
        [&destination_space],
        |key, indices| u1_value(key, indices, 10_000.0),
    )
    .unwrap();
    let mut source_shapes = HashMap::<BlockKey, Vec<usize>>::new();
    let source = Tensor::from_block_fn(
        &runtime,
        [&source_space],
        [&source_space],
        |key, indices| {
            let shape = source_shapes
                .entry(key.clone())
                .or_insert_with(|| vec![0; indices.len()]);
            for (extent, &index) in shape.iter_mut().zip(indices) {
                *extent = (*extent).max(index + 1);
            }
            u1_value(key, indices, 20_000.0)
        },
    )
    .unwrap();
    let destination_before = destination.data().to_vec();
    let source_before = source.data().to_vec();

    let actual = destination.absorb(&source).unwrap();
    let expected = Tensor::from_block_fn(
        &runtime,
        [&destination_space],
        [&destination_space],
        |key, indices| {
            if source_shapes.get(key).is_some_and(|shape| {
                indices
                    .iter()
                    .zip(shape)
                    .all(|(&index, &extent)| index < extent)
            }) {
                u1_value(key, indices, 20_000.0)
            } else {
                u1_value(key, indices, 10_000.0)
            }
        },
    )
    .unwrap();

    assert_eq!(actual.data(), expected.data());
    assert_eq!(actual.dtype(), Dtype::F64);
    assert_eq!(actual.codomain_spaces(), destination.codomain_spaces());
    assert_eq!(actual.domain_spaces(), destination.domain_spaces());
    assert_eq!(destination.data(), destination_before);
    assert_eq!(source.data(), source_before);
    assert_eq!(destination.storage_strong_count(), 1);
    assert_eq!(actual.storage_strong_count(), 1);
}

#[test]
fn absorb_matches_frozen_tensorkit_column_major_oracle() {
    // What: TensorKit absorb! on a 3x4 destination and 2x5 source overwrites
    // exactly the 2x4 top-left prefix in column-major reduced storage.
    let runtime = Runtime::builder().build().unwrap();
    let destination_codomain = Space::u1([(0, 3)]);
    let destination_domain = Space::u1([(0, 4)]);
    let source_codomain = Space::u1([(0, 2)]);
    let source_domain = Space::u1([(0, 5)]);
    let destination = Tensor::from_block_fn(
        &runtime,
        [&destination_codomain],
        [&destination_domain],
        |_, indices| 100.0 + indices[0] as f64 + 10.0 * indices[1] as f64,
    )
    .unwrap();
    let source = Tensor::from_block_fn(
        &runtime,
        [&source_codomain],
        [&source_domain],
        |_, indices| indices[0] as f64 + 10.0 * indices[1] as f64,
    )
    .unwrap();

    assert_eq!(
        destination.absorb(&source).unwrap().data(),
        &[0.0, 1.0, 102.0, 10.0, 11.0, 112.0, 20.0, 21.0, 122.0, 30.0, 31.0, 132.0]
    );
}

#[test]
fn absorb_scalar_conversion_is_overlap_local_and_exact() {
    let runtime = Runtime::builder().build().unwrap();
    let one = Space::u1([(0, 1)]);
    let two = Space::u1([(0, 2)]);
    let three = Space::u1([(0, 3)]);

    let real_destination = Tensor::from_block_fn(&runtime, [&two], [&one], |_, _| -1.0).unwrap();
    let accepted =
        Tensor::from_block_fn(&runtime, [&three], [&one], |_, indices| match indices[0] {
            0 => Complex64::new(f64::NAN, 0.0),
            1 => Complex64::new(4.0, -0.0),
            _ => Complex64::new(3.0, 7.0),
        })
        .unwrap();
    let output = real_destination.absorb(&accepted).unwrap();
    assert!(output.data()[0].is_nan());
    assert_eq!(output.data()[1], 4.0);

    for imaginary in [1.0, f64::NAN] {
        let rejected = Tensor::from_block_fn(&runtime, [&three], [&one], |_, indices| {
            Complex64::new(2.0, if indices[0] == 0 { imaginary } else { 0.0 })
        })
        .unwrap();
        let destination_before = real_destination.data().to_vec();
        let source_before = rejected.data_c64().to_vec();
        assert!(matches!(
            real_destination.absorb(&rejected),
            Err(Error::InexactScalarConversion {
                operation: "Tensor::absorb",
                from: Dtype::C64,
                to: Dtype::F64,
            })
        ));
        assert_eq!(real_destination.data(), destination_before);
        assert!(rejected
            .data_c64()
            .iter()
            .zip(source_before)
            .all(
                |(actual, expected)| actual.re.to_bits() == expected.re.to_bits()
                    && actual.im.to_bits() == expected.im.to_bits()
            ));
    }

    let complex_destination =
        Tensor::from_block_fn(&runtime, [&one], [&one], |_, _| Complex64::new(-1.0, 4.0)).unwrap();
    let real_source = Tensor::from_block_fn(&runtime, [&one], [&one], |_, _| 6.0).unwrap();
    assert_eq!(
        complex_destination.absorb(&real_source).unwrap().data_c64(),
        &[Complex64::new(6.0, 0.0)]
    );
}

#[test]
fn absorb_empty_overlap_and_rank_zero_are_total() {
    let runtime = Runtime::builder().build().unwrap();
    let zero = Space::u1([(0, 2)]);
    let one = Space::u1([(1, 3)]);
    let destination = Tensor::from_block_fn(&runtime, [&zero], [&zero], |_, indices| {
        (indices[0] + 10 * indices[1]) as f64
    })
    .unwrap();
    let source = Tensor::from_block_fn(&runtime, [&one], [&one], |_, _| {
        Complex64::new(1.0, f64::NAN)
    })
    .unwrap();
    assert_eq!(
        destination.absorb(&source).unwrap().data(),
        destination.data()
    );

    let zero_extent = Space::u1([(0, 0)]);
    let empty = Tensor::zeros(&runtime, Dtype::F64, [&zero_extent], [&zero_extent]).unwrap();
    let nonreal = Tensor::from_block_fn(&runtime, [&zero], [&zero], |_, _| {
        Complex64::new(1.0, f64::NAN)
    })
    .unwrap();
    assert!(empty.absorb(&nonreal).unwrap().data().is_empty());

    let vector = Tensor::from_block_fn(&runtime, [&zero], [], |_, _| 2.0).unwrap();
    let covector = Tensor::from_block_fn(&runtime, [], [&zero], |_, _| 3.0).unwrap();
    let source_vector = Tensor::from_block_fn(&runtime, [&zero], [], |_, _| 5.0).unwrap();
    let source_covector = Tensor::from_block_fn(&runtime, [], [&zero], |_, _| 7.0).unwrap();
    let scalar_destination = covector.compose(&vector).unwrap();
    let scalar_source = source_covector.compose(&source_vector).unwrap();
    let scalar = scalar_destination.absorb(&scalar_source).unwrap();
    assert_eq!(scalar.rank(), 0);
    assert_eq!(scalar.scalar().unwrap(), scalar_source.scalar().unwrap());
}

#[test]
fn absorb_uses_complete_tree_identity_across_supported_fusion_styles() {
    fn check(runtime: &Runtime, space: &Space) {
        let destination =
            Tensor::from_block_fn(runtime, [space, space], [space], |_, _| -1.0).unwrap();
        let source = Tensor::from_block_fn(runtime, [space, space], [space], |key, _| {
            let BlockKey::FusionTree(key) = key else {
                unreachable!()
            };
            key.codomain_tree()
                .vertices()
                .iter()
                .chain(key.domain_tree().vertices())
                .enumerate()
                .map(|(position, vertex)| vertex.get() as f64 * 10f64.powi(position as i32))
                .sum::<f64>()
                + key.codomain_tree().innerlines().len() as f64 * 1_000.0
        })
        .unwrap();
        assert_eq!(destination.absorb(&source).unwrap().data(), source.data());
    }

    let runtime = Runtime::builder().build().unwrap();
    check(&runtime, &Space::su2([(0, 1), (1, 1), (2, 1)]).unwrap());
    check(&runtime, &Space::fz2([(0, 1), (1, 1)]).unwrap());
    check(
        &runtime,
        &Space::product([((0, 0), 1), ((1, 1), 1)]).unwrap(),
    );
    check(&runtime, &Space::su3([((1, 1), 1)]).unwrap());
}

#[test]
fn absorb_validation_precedence_and_duality_are_stable() {
    let runtime = Runtime::builder().build().unwrap();
    let other_runtime = Runtime::builder().build().unwrap();
    let u1 = Space::u1([(0, 1)]);
    let z2 = Space::z2([(0, 1)]);
    let destination = Tensor::zeros(&runtime, Dtype::F64, [&u1], [&u1]).unwrap();
    let bad_rank = Tensor::zeros(&other_runtime, Dtype::F64, [&z2, &z2], [&z2]).unwrap();
    // What: rank/split validation wins when the fusion rule is also wrong.
    assert!(matches!(
        destination.absorb(&bad_rank),
        Err(Error::InvalidArgument(_))
    ));

    let bad_rule = Tensor::zeros(&other_runtime, Dtype::F64, [&z2], [&z2]).unwrap();
    // What: rule validation wins when the runtime is also wrong.
    assert!(matches!(
        destination.absorb(&bad_rule),
        Err(Error::RuleMismatch)
    ));

    let bad_runtime = Tensor::zeros(&other_runtime, Dtype::F64, [&u1], [&u1]).unwrap();
    assert!(matches!(
        destination.absorb(&bad_runtime),
        Err(Error::RuntimeMismatch)
    ));

    let dual = u1.dual();
    let dual_mismatch =
        Tensor::from_block_fn(&runtime, [&dual], [&dual], |_, _| Complex64::new(1.0, 2.0)).unwrap();
    // What: duality validation wins over an in-overlap inexact conversion.
    assert!(matches!(
        destination.absorb(&dual_mismatch),
        Err(Error::InvalidArgument(message)) if message.contains("duality")
    ));
}

#[test]
fn absorb_materializes_lazy_operands() {
    let runtime = Runtime::builder().build().unwrap();
    let space = Space::u1([(-1, 1), (0, 2), (1, 1)]);
    let destination_parent =
        Tensor::rand_with_seed(&runtime, Dtype::C64, [&space], [&space], 395_001).unwrap();
    let source_parent =
        Tensor::rand_with_seed(&runtime, Dtype::C64, [&space], [&space], 395_002).unwrap();
    let destination = destination_parent.adjoint().unwrap();
    let source = source_parent.adjoint().unwrap();

    assert_eq!(
        destination.absorb(&source).unwrap().data_c64(),
        source.data_c64()
    );
}
