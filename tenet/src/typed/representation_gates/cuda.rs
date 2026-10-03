use super::*;

#[cfg(feature = "cuda")]
fn assert_cuda_tensor_matches_host<R>(
    actual: &TensorMap<R, f64>,
    expected: &TensorMap<R, f64>,
    provider: *const R,
    runtime: &crate::runtime::RuntimeIdentity,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    assert!(std::ptr::eq(actual.provider(), provider));
    assert!(runtime.matches(actual.runtime()));
    assert_eq!(
        actual.logical_space().space(),
        expected.logical_space().space()
    );
    assert_eq!(actual.dense_data().unwrap(), expected.dense_data().unwrap());
    assert_eq!(actual.subblock_count(), expected.subblock_count());
    for index in 0..actual.subblock_count() {
        let actual_block = actual.subblock(index).unwrap();
        let expected_block = expected.subblock(index).unwrap();
        assert_eq!(actual_block.key(), expected_block.key());
        assert_eq!(actual_block.offset(), expected_block.offset());
        assert_eq!(actual_block.shape(), expected_block.shape());
        assert_eq!(actual_block.strides(), expected_block.strides());
        assert_eq!(
            actual.subblock_fusion_trees(index).unwrap(),
            expected.subblock_fusion_trees(index).unwrap()
        );
    }
}

#[cfg(feature = "cuda")]
fn assert_cuda_lazy_contract_orientations<R>(lhs: &TensorMap<R, f64>, rhs: &TensorMap<R, f64>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    let lhs_axes: Vec<_> = (lhs.codomain_rank()..lhs.rank()).collect();
    let rhs_axes: Vec<_> = (0..rhs.codomain_rank()).collect();
    let output_axes: Vec<_> = (0..lhs.codomain_rank() + rhs.domain_rank()).collect();
    let (codomain, domain) = output_axes.split_at(lhs.codomain_rank());
    let spec = ContractSpec {
        lhs: &lhs_axes,
        rhs: &rhs_axes,
        codomain,
        domain,
    };
    let expected_contract = lhs.contract(rhs, &spec).unwrap();
    let expected_compose = lhs.compose(rhs).unwrap();
    let provider = lhs.provider() as *const R;
    let runtime = lhs.runtime().identity();

    for (lhs_adjoint, rhs_adjoint) in [(false, false), (true, false), (false, true), (true, true)] {
        let device_operand = |logical: &TensorMap<R, f64>, adjoint: bool| {
            if adjoint {
                eager_adjoint_oracle(logical)
                    .to_cuda()
                    .unwrap()
                    .adjoint()
                    .unwrap()
            } else {
                logical.to_cuda().unwrap()
            }
        };
        let lhs_device = device_operand(lhs, lhs_adjoint);
        let rhs_device = device_operand(rhs, rhs_adjoint);
        let contract = lhs_device
            .contract(&rhs_device, &spec)
            .unwrap()
            .to_host()
            .unwrap();
        let compose = lhs_device.compose(&rhs_device).unwrap().to_host().unwrap();

        assert_cuda_tensor_matches_host(&contract, &expected_contract, provider, &runtime);
        assert_cuda_tensor_matches_host(&compose, &expected_compose, provider, &runtime);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn typed_cuda_owned_metadata_validation_orders_ordinal_before_length() {
    type DeviceTensor = TensorMap<U1FusionRule, f64, CudaStorage>;
    assert!(DeviceTensor::validate_cuda_owned_metadata(
        Placement::Cuda(0),
        Placement::Cuda(0),
        7,
        7
    )
    .is_ok());
    assert_eq!(
        DeviceTensor::validate_cuda_owned_metadata(Placement::Cuda(1), Placement::Cuda(0), 7, 6)
            .unwrap_err(),
        Error::PlacementMismatch
    );
    assert!(matches!(
        DeviceTensor::validate_cuda_owned_metadata(
            Placement::Cuda(0),
            Placement::Cuda(0),
            7,
            6
        ),
        Err(Error::InvalidArgument(message)) if message.contains("payload length")
    ));
}

#[cfg(feature = "cuda")]
#[test]
fn missing_cuda_context_precedes_compact_expansion_and_lazy_materialization() {
    DIAGONAL_MATERIALIZATIONS.set(0);
    let source = u1_lazy_fixture();
    let diagonal = source
        .svd_compact(&codomain_axes(&source), &domain_axes(&source))
        .unwrap()
        .s;
    let lazy = source.adjoint().unwrap();
    let TypedData::Diagonal(spectrum) = owned(&diagonal).data.as_ref() else {
        unreachable!("SVD factor is compact")
    };
    let mut malformed_spectrum = spectrum.clone();
    for entry in &mut malformed_spectrum {
        entry.values.clear();
    }
    let malformed_expansion = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        tenet_matrixalgebra::seam::diagonal_bond_data(
            diagonal.logical_space().space(),
            &malformed_spectrum,
            &|value| value,
        )
    }));
    assert!(
        malformed_expansion.is_err() || matches!(malformed_expansion, Ok(Err(_))),
        "the fixture must fail if compact expansion runs"
    );
    let malformed = TensorMap {
        runtime: diagonal.runtime.clone(),
        repr: owned_repr(TypedTensorBody::diagonal(
            diagonal.logical_space().clone(),
            malformed_spectrum,
        )),
    };
    let missing_context = Error::InvalidArgument(
        "this runtime was built without a CUDA device; use Runtime::builder().cuda(device)"
            .to_string(),
    );

    assert_eq!(malformed.to_cuda().unwrap_err(), missing_context);
    assert!(matches!(lazy.to_cuda(), Err(error) if error == missing_context));
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn typed_cuda_compact_and_lazy_roundtrips_keep_source_caches_cold() {
    DIAGONAL_MATERIALIZATIONS.set(0);
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();

    let diagonal = source
        .svd_compact(&codomain_axes(&source), &domain_axes(&source))
        .unwrap()
        .s;
    let TypedData::Diagonal(spectrum) = owned(&diagonal).data.as_ref() else {
        unreachable!("SVD factor is compact")
    };
    let expected_diagonal = tenet_matrixalgebra::seam::diagonal_bond_data(
        diagonal.logical_space().space(),
        spectrum,
        &|value| value,
    )
    .unwrap();
    let diagonal_device = diagonal.to_cuda().unwrap();
    assert_eq!(diagonal_device.placement(), Placement::Cuda(0));
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    let diagonal_host = diagonal_device.to_host().unwrap();
    assert!(matches!(
        owned(&diagonal_host).data.as_ref(),
        TypedData::Dense(_)
    ));
    assert_eq!(diagonal_host.dense_data().unwrap(), expected_diagonal);
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);

    let lazy = source.adjoint().unwrap();
    let expected_lazy = tenet_tensors::materialize_adjoint_data_dyn(
        source.logical_space().space(),
        lazy.logical_space().space(),
        source.dense_data().unwrap(),
    )
    .unwrap();
    let lazy_device = lazy.to_cuda().unwrap();
    let TypedTensorRepr::Adjoint(device_view) = &lazy_device.repr else {
        unreachable!("transfer preserves the lazy view")
    };
    let expected_norm = source.norm(2.0).unwrap();
    CUDA_REDUCTION_BUFFER_OBSERVATION.with(|observation| observation.set(Some((0, 0, 0))));
    assert!(
        (lazy_device.norm(2.0).unwrap() - expected_norm).abs() <= 1e-12 * (1.0 + expected_norm)
    );
    let observed = CUDA_REDUCTION_BUFFER_OBSERVATION.with(|observation| {
        let observed = observation.get().unwrap();
        observation.set(None);
        observed
    });
    let sector_count = sector_regions(
        source.logical_space().space().structure(),
        source.logical_space().space().nout(),
    )
    .unwrap()
    .len();
    assert_eq!(observed, (1, sector_count.max(1), sector_count.max(1)));
    assert!(source.dense_data().unwrap().len() > sector_count.max(1));

    macro_rules! observed_arithmetic {
        ($expression:expr, $arithmetic:expr, $reduction:expr) => {{
            CUDA_ARITHMETIC_OBSERVATION.with(|observation| observation.set(Some((0, 0, 0))));
            CUDA_REDUCTION_BUFFER_OBSERVATION.with(|observation| observation.set(Some((0, 0, 0))));
            let result = $expression;
            CUDA_ARITHMETIC_OBSERVATION.with(|observation| {
                assert_eq!(observation.get(), Some($arithmetic));
                observation.set(None);
            });
            CUDA_REDUCTION_BUFFER_OBSERVATION.with(|observation| {
                assert_eq!(observation.get(), Some($reduction));
                observation.set(None);
            });
            result
        }};
    }

    let source_device = source.to_cuda().unwrap();
    let empty_storage = {
        let lease = runtime.lease_cuda().unwrap();
        CudaStorage::<f64>::upload(&lease, &[]).unwrap()
    };
    let malformed_length = TensorMap {
        runtime: runtime.clone(),
        repr: owned_repr(TypedTensorBody::dense(
            source.logical_space().clone(),
            empty_storage,
        )),
    };
    let work_sentinel = (usize::MAX, usize::MAX, usize::MAX);
    CUDA_ARITHMETIC_OBSERVATION.with(|observation| observation.set(Some(work_sentinel)));
    assert!(matches!(
        malformed_length.scale(2.0),
        Err(Error::InvalidArgument(message)) if message.contains("payload length")
    ));
    CUDA_ARITHMETIC_OBSERVATION.with(|observation| {
        assert_eq!(observation.get(), Some(work_sentinel));
        observation.set(None);
    });

    observed_arithmetic!(source_device.scale(-2.0), (1, 1, 1), (0, 0, 0)).unwrap();
    observed_arithmetic!(
        source_device.axpby(2.0, &source_device, -3.0),
        (1, 1, 2),
        (0, 0, 0)
    )
    .unwrap();
    observed_arithmetic!(source_device.zeros_like(), (1, 0, 0), (0, 0, 0)).unwrap();

    let lazy_scale = observed_arithmetic!(lazy_device.scale(-2.0), (1, 1, 1), (0, 0, 0)).unwrap();
    let lazy_add = observed_arithmetic!(
        lazy_device.axpby(2.0, &lazy_device, -3.0),
        (1, 1, 2),
        (0, 0, 0)
    )
    .unwrap();
    let lazy_zero = observed_arithmetic!(lazy_device.zeros_like(), (1, 0, 0), (0, 0, 0)).unwrap();
    for result in [&lazy_scale, &lazy_add, &lazy_zero] {
        assert!(matches!(result.repr, TypedTensorRepr::Adjoint(_)));
    }
    assert!(matches!(
        observed_arithmetic!(
            lazy_device.axpby(2.0, &source_device, -3.0),
            (0, 0, 0),
            (0, 0, 0)
        ),
        Err(Error::UnsupportedOnDevice(_))
    ));

    assert!(matches!(
        lazy_device.inner(&lazy_device),
        Err(Error::UnsupportedOnDevice(_))
    ));
    assert!(matches!(
        lazy_device.inner(&lazy_device),
        Err(Error::UnsupportedOnDevice(_))
    ));

    let mut missing_context = lazy_device.clone();
    missing_context.runtime = Runtime::builder().build().unwrap();
    let preflight_sentinel = (usize::MAX, usize::MAX, usize::MAX);
    CUDA_REDUCTION_BUFFER_OBSERVATION.with(|observation| observation.set(Some(preflight_sentinel)));
    assert!(matches!(
        missing_context.norm(2.0),
        Err(Error::InvalidArgument(message)) if message.contains("without a CUDA device")
    ));
    CUDA_REDUCTION_BUFFER_OBSERVATION.with(|observation| {
        assert_eq!(observation.get(), Some(preflight_sentinel));
        observation.set(None);
    });
    CUDA_ARITHMETIC_OBSERVATION.with(|observation| observation.set(Some(preflight_sentinel)));
    assert!(matches!(
        missing_context.zeros_like(),
        Err(Error::InvalidArgument(message)) if message.contains("without a CUDA device")
    ));
    CUDA_ARITHMETIC_OBSERVATION.with(|observation| {
        assert_eq!(observation.get(), Some(preflight_sentinel));
        observation.set(None);
    });
    let device_clone = lazy_device.clone();
    let TypedTensorRepr::Adjoint(clone_view) = &device_clone.repr else {
        unreachable!("clone preserves the lazy view")
    };
    assert!(Arc::ptr_eq(device_view, clone_view));

    let lazy_host = device_clone.to_host().unwrap();
    let TypedTensorRepr::Adjoint(_) = &lazy_host.repr else {
        unreachable!("roundtrip preserves the lazy view")
    };
    assert_eq!(
        lazy_host.materialize().unwrap().dense_data().unwrap(),
        expected_lazy
    );
}

/// #1268: a `Complex64` device payload must perform exactly the same
/// number of device allocations, coefficient uploads, kernels, and
/// reduction downloads as the `f64` payload on the same structure. Only
/// the bytes per element change (covered in `tenet-dense`).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_complex_payload_costs_the_same_device_calls_as_f64() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let real: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
    let complex: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
            let ramp = indices.iter().sum::<usize>() as f64;
            Complex64::new(ramp + 1.0, -(ramp + 1.75))
        })
        .unwrap();

    fn observe<T>(run: impl FnOnce() -> T) -> ((usize, usize, usize), (usize, usize, usize)) {
        CUDA_ARITHMETIC_OBSERVATION.with(|observation| observation.set(Some((0, 0, 0))));
        CUDA_REDUCTION_BUFFER_OBSERVATION.with(|observation| observation.set(Some((0, 0, 0))));
        drop(run());
        let arithmetic = CUDA_ARITHMETIC_OBSERVATION.with(|observation| {
            let observed = observation.get().unwrap();
            observation.set(None);
            observed
        });
        let reduction = CUDA_REDUCTION_BUFFER_OBSERVATION.with(|observation| {
            let observed = observation.get().unwrap();
            observation.set(None);
            observed
        });
        (arithmetic, reduction)
    }

    let real_device = real.to_cuda().unwrap();
    let complex_device = complex.to_cuda().unwrap();
    assert_eq!(
        observe(|| real_device.scale(-2.0).unwrap()),
        observe(|| complex_device.scale(Complex64::new(-2.0, 0.5)).unwrap())
    );
    assert_eq!(
        observe(|| real_device.axpby(2.0, &real_device, -3.0).unwrap()),
        observe(|| complex_device
            .axpby(
                Complex64::new(2.0, 1.0),
                &complex_device,
                Complex64::new(-3.0, 0.25)
            )
            .unwrap())
    );
    assert_eq!(
        observe(|| real_device.zeros_like().unwrap()),
        observe(|| complex_device.zeros_like().unwrap())
    );
    assert_eq!(
        observe(|| real_device.inner(&real_device).unwrap()),
        observe(|| complex_device.inner(&complex_device).unwrap())
    );
    // Contraction allocates its destination and runs its kernels inside
    // the replay seam, which has no arithmetic/reduction hooks: both
    // dtypes must leave those counters untouched rather than falling back
    // to the axpby or reduction paths.
    let untouched = ((0, 0, 0), (0, 0, 0));
    assert_eq!(
        observe(|| real_device
            .contract(
                &real_device,
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[0],
                    domain: &[1]
                }
            )
            .unwrap()),
        untouched
    );
    assert_eq!(
        observe(|| complex_device
            .contract(
                &complex_device,
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[0],
                    domain: &[1]
                }
            )
            .unwrap()),
        untouched
    );
}

#[cfg(feature = "cuda")]
#[test]
fn typed_cuda_reduction_placement_validation_is_exact() {
    assert!(validate_cuda_reduction_placement(
        Placement::Cuda(0),
        Placement::Cuda(0),
        Placement::Cuda(0)
    )
    .is_ok());
    assert_eq!(
        validate_cuda_reduction_placement(
            Placement::Cuda(1),
            Placement::Cuda(0),
            Placement::Cuda(0)
        )
        .unwrap_err(),
        Error::PlacementMismatch
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_lazy_adjoint_contract_and_compose_match_rectangular_host_oracles() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = |degeneracy| {
        GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), degeneracy)]).unwrap()
    };
    let (m, k, n) = (2, 3, 4);
    let lhs = TensorMap::from_subblock_fn(&runtime, [&leg(m)], [&leg(k)], |_, indices| {
        (indices[0] + m * indices[1]) as f64 + 1.0
    })
    .unwrap();
    let rhs = TensorMap::from_subblock_fn(&runtime, [&leg(k)], [&leg(n)], |_, indices| {
        (2 * indices[0] + indices[1]) as f64 + 1.0
    })
    .unwrap();
    let expected_contract = lhs
        .contract(
            &rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let expected_compose = lhs.compose(&rhs).unwrap();

    for upload_parent_first in [false, true] {
        for (lhs_adjoint, rhs_adjoint) in
            [(false, false), (true, false), (false, true), (true, true)]
        {
            let device_operand = |logical: &TensorMap<U1FusionRule, f64>, adjoint: bool| {
                if !adjoint {
                    return logical.to_cuda().unwrap();
                }
                let parent = eager_adjoint_oracle(logical);
                if upload_parent_first {
                    parent.to_cuda().unwrap().adjoint().unwrap()
                } else {
                    parent.adjoint().unwrap().to_cuda().unwrap()
                }
            };
            let lhs_device = device_operand(&lhs, lhs_adjoint);
            let rhs_device = device_operand(&rhs, rhs_adjoint);
            let contracted = lhs_device
                .contract(
                    &rhs_device,
                    &ContractSpec {
                        lhs: &[1],
                        rhs: &[0],
                        codomain: &[0],
                        domain: &[1],
                    },
                )
                .unwrap();
            let composed = lhs_device.compose(&rhs_device).unwrap();
            let contracted = contracted.to_host().unwrap();
            let composed = composed.to_host().unwrap();

            assert_eq!(
                contracted.logical_space().space(),
                expected_contract.logical_space().space()
            );
            assert_eq!(
                composed.logical_space().space(),
                expected_compose.logical_space().space()
            );
            assert_eq!(
                contracted.dense_data().unwrap(),
                expected_contract.dense_data().unwrap()
            );
            assert_eq!(
                composed.dense_data().unwrap(),
                expected_compose.dense_data().unwrap()
            );
            assert!(Arc::ptr_eq(
                contracted.logical_space().provider_arc(),
                lhs.logical_space().provider_arc()
            ));
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_lazy_adjoint_preserves_fermionic_contract_sign() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let provider = Arc::new(FermionParityFusionRule);
    let odd = |is_dual| {
        GradedSpace::try_new(Arc::clone(&provider), [(Z2Irrep::ODD, 1)])
            .and_then(|space| if is_dual { space.try_dual() } else { Ok(space) })
            .unwrap()
    };
    let lhs =
        TensorMap::from_subblock_fn(&runtime, [&odd(false)], [&odd(true)], |_, _| 2.0).unwrap();
    let rhs =
        TensorMap::from_subblock_fn(&runtime, [&odd(true)], [&odd(false)], |_, _| 3.0).unwrap();

    for (lhs_adjoint, rhs_adjoint) in [(false, false), (true, false), (false, true), (true, true)] {
        let device_operand = |logical: &TensorMap<FermionParityFusionRule, f64>, adjoint: bool| {
            if adjoint {
                eager_adjoint_oracle(logical)
                    .to_cuda()
                    .unwrap()
                    .adjoint()
                    .unwrap()
            } else {
                logical.to_cuda().unwrap()
            }
        };
        let lhs_device = device_operand(&lhs, lhs_adjoint);
        let rhs_device = device_operand(&rhs, rhs_adjoint);
        let contract = lhs_device
            .contract(
                &rhs_device,
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[0],
                    domain: &[1],
                },
            )
            .unwrap()
            .to_host()
            .unwrap();
        let compose = lhs_device.compose(&rhs_device).unwrap().to_host().unwrap();

        assert_eq!(contract.dense_data().unwrap(), &[-6.0]);
        assert_eq!(compose.dense_data().unwrap(), &[6.0]);
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_lazy_adjoint_covers_su2_rank_five_and_simple_product() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();

    let su2_provider = Arc::new(SU2FusionRule);
    let su2 = GradedSpace::try_new(
        Arc::clone(&su2_provider),
        [
            (SU2Irrep::from_twice_spin(0), 1),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();
    let su2_lhs =
        TensorMap::from_subblock_fn(&runtime, [&su2, &su2, &su2], [&su2, &su2], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
    let su2_rhs =
        TensorMap::from_subblock_fn(&runtime, [&su2, &su2], [&su2, &su2, &su2], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 3.0
        })
        .unwrap();
    assert_cuda_lazy_contract_orientations(&su2_lhs, &su2_rhs);

    let product_provider = Arc::new(U1FusionRule.product(FermionParityFusionRule));
    let product = GradedSpace::try_new(
        Arc::clone(&product_provider),
        [
            (product_sector(U1Irrep::new(0), Z2Irrep::EVEN), 2),
            (product_sector(U1Irrep::new(1), Z2Irrep::ODD), 1),
        ],
    )
    .unwrap();
    let product_lhs =
        TensorMap::from_subblock_fn(&runtime, [&product], [&product], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
    let product_rhs =
        TensorMap::from_subblock_fn(&runtime, [&product], [&product], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 4.0
        })
        .unwrap();
    assert_cuda_lazy_contract_orientations(&product_lhs, &product_rhs);
}

/// GL-3 (#1281): a device operation must not need the coarse Runtime
/// state mutex. The reverse direction is the observable one: Host
/// standalone contraction never takes `state`, so only a parked holder of
/// that lock can prove a device operation is independent of it.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn device_work_completes_while_another_thread_holds_the_runtime_state_lock() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;
    use std::sync::Barrier;
    use std::time::Duration;

    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    let lhs = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        indices[0] as f64 + 1.0
    })
    .unwrap();
    let rhs = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        indices[1] as f64 + 2.0
    })
    .unwrap();

    // Independent oracle for the device result, computed on Host before
    // the state lock is parked.
    let host_expected = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        indices[0] as f64 + 1.0
    })
    .unwrap()
    .contract(
        &TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
            indices[1] as f64 + 2.0
        })
        .unwrap(),
        &ContractSpec {
            lhs: &[1],
            rhs: &[0],
            codomain: &[0],
            domain: &[1],
        },
    )
    .unwrap();

    let holding = Arc::new(Barrier::new(2));
    let release = Arc::new(AtomicBool::new(false));
    let (sender, receiver) = mpsc::channel();

    let values = std::thread::scope(|scope| {
        let holder_runtime = runtime.clone();
        let holder_barrier = Arc::clone(&holding);
        let holder_release = Arc::clone(&release);
        scope.spawn(move || {
            let _state = holder_runtime.lock();
            holder_barrier.wait();
            while !holder_release.load(Ordering::SeqCst) {
                std::thread::yield_now();
            }
        });

        holding.wait();
        scope.spawn(move || {
            let device = lhs
                .to_cuda()
                .and_then(|lhs| Ok((lhs, rhs.to_cuda()?)))
                .and_then(|(lhs, rhs)| {
                    lhs.contract(
                        &rhs,
                        &ContractSpec {
                            lhs: &[1],
                            rhs: &[0],
                            codomain: &[0],
                            domain: &[1],
                        },
                    )
                })
                .and_then(|out| out.to_host());
            let _ = sender.send(device.map(|out| out.dense_data().unwrap().to_vec()));
        });

        let outcome = receiver.recv_timeout(Duration::from_secs(30));
        release.store(true, Ordering::SeqCst);
        outcome
            .expect("device transfer and contraction blocked on the Runtime state lock")
            .expect("device contraction failed")
    });

    // The device result under the parked state lock is the Host result.
    assert_eq!(values.len(), host_expected.dense_data().unwrap().len());
    assert!(values.iter().any(|value| *value != 0.0));
    for (actual, expected) in values.iter().zip(host_expected.dense_data().unwrap()) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "device value {actual} differs from the Host oracle {expected}"
        );
    }
}

/// GL-3 (#1281): the device lock nests with the CPU leases in either
/// order without deadlocking, because it is a separate mutex.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn device_lease_nests_with_cpu_leases_in_both_orders() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();

    {
        let cuda = runtime.lease_cuda().unwrap();
        let context = runtime.lease_context().unwrap();
        let dense = runtime.lease_dense();
        drop(dense);
        // A CPU lease released under the device guard must not deadlock.
        drop(context);
        drop(cuda);
    }

    let context = runtime.lease_context().unwrap();
    let cuda = runtime.lease_cuda().unwrap();
    drop(context);
    drop(cuda);
}

/// GL-3 (#1281): device lowering touches no execution context, so a
/// device contraction leaves the Runtime tree-transform cache untouched.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn device_contraction_leaves_the_tree_transform_cache_unchanged() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    let lhs = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        indices[0] as f64 + 1.0
    })
    .unwrap()
    .to_cuda()
    .unwrap();
    let rhs = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        indices[1] as f64 + 2.0
    })
    .unwrap()
    .to_cuda()
    .unwrap();

    let before = runtime.tree_transform_cache_info().structures;
    let product = lhs
        .contract(
            &rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    assert_eq!(product.placement(), Placement::Cuda(0));
    assert_eq!(runtime.tree_transform_cache_info().structures, before);
}

/// A destination whose device is not the Runtime's is rejected before any
/// device work, its bytes untouched (#1551). The public API cannot build
/// one — a tensor's storage lives on its own Runtime's device, and the
/// Runtime check comes first — so this in-crate gate forges it from a
/// second device's tensor.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires two real CUDA devices"]
fn typed_cuda_into_rejects_a_foreign_device_destination_untouched() {
    let rt0 = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let rt1 = Runtime::builder().cuda(1).dense_threads(1).build().unwrap();
    let v = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(0), 2),
            (U1Irrep::new(1), 1),
            (U1Irrep::new(-1), 2),
        ],
    )
    .unwrap();
    let w = v.try_dual().unwrap();
    let host_t = TensorMap::<_, f64>::rand_with_seed(&rt0, [&v, &w], [&v, &w], 1).unwrap();
    let host_rhs = TensorMap::<_, f64>::rand_with_seed(&rt0, [&v, &w], [&v], 2).unwrap();
    let (t, rhs) = (host_t.to_cuda().unwrap(), host_rhs.to_cuda().unwrap());
    let spec = ContractSpec {
        lhs: &[2, 3],
        rhs: &[0, 1],
        codomain: &[2, 0],
        domain: &[1],
    };
    let foreign = |like: TensorMap<U1FusionRule, f64>| {
        let (codomain, domain) = (like.codomain(), like.domain());
        let mut next = 0u64;
        let mut device = TensorMap::<_, f64>::from_subblock_fn(&rt1, &codomain, &domain, |_, _| {
            next += 1;
            std::f64::consts::PI * next as f64
        })
        .unwrap()
        .to_cuda()
        .unwrap();
        let before: Vec<u64> = device
            .to_host()
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect();
        device.runtime = rt0.clone();
        (device, before)
    };
    type Device = TensorMap<U1FusionRule, f64, CudaStorage<f64>>;
    let check = |what: &str,
                 (mut device, before): (Device, Vec<u64>),
                 call: &dyn Fn(&mut Device) -> Result<(), Error>| {
        assert_eq!(call(&mut device), Err(Error::PlacementMismatch), "{what}");
        device.runtime = rt1.clone();
        let after: Vec<u64> = device
            .to_host()
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect();
        assert_eq!(after, before, "{what}: destination changed");
    };
    check(
        "permute_into",
        foreign(host_t.permute(&[2, 0], &[1, 3]).unwrap()),
        &|d| t.permute_into(&[2, 0], &[1, 3], d, 1.0, 0.5),
    );
    check(
        "braid_into",
        foreign(host_t.braid(&[1, 0], &[3, 2], &[0, 1, 2, 3]).unwrap()),
        &|d| t.braid_into(&[1, 0], &[3, 2], &[0, 1, 2, 3], d, 1.0, 0.5),
    );
    check(
        "transpose_into",
        foreign(host_t.transpose(&[1, 3], &[0, 2]).unwrap()),
        &|d| t.transpose_into(&[1, 3], &[0, 2], d, 1.0, 0.5),
    );
    check(
        "repartition_into",
        foreign(host_t.repartition(1).unwrap()),
        &|d| t.repartition_into(d, 1.0, 0.5),
    );
    check(
        "trace_pairs_into",
        foreign(host_t.trace_pairs(&[(0, 2)]).unwrap()),
        &|d| t.trace_pairs_into(&[(0, 2)], d, 1.0, 0.5),
    );
    check(
        "contract_into",
        foreign(host_t.contract(&host_rhs, &spec).unwrap()),
        &|d| t.contract_into(&rhs, &spec, d, 1.0, 0.5),
    );
    check("axpby_into", foreign(host_t.clone()), &|d| {
        t.axpby_into(d, 1.0, 0.5)
    });
}
