use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use tenet::core::{
    BraidingStyleKind, CheckedGenericAdmissionMode, CheckedGenericFusion,
    CheckedGenericRigidSymbols, CheckedGenericStructureError, FusionStyleKind, GenericFArray,
    GenericRMatrix, RuleIdentity, SectorId, SectorVec, TypedSectorAdmission,
};
use tenet::dense::{
    DefaultDenseExecutor, DenseBackend, DenseDotConfig, DenseError, DenseExecutor, DenseRead,
    DenseTensor, DenseWrite,
};
use tenet::prelude::{Complex64, GenericTensorError, Runtime, SectorSpectrum};
use tenet::typed::{
    CheckedGenericTensorProductError, GradedSpace, NetworkReuseClass, TensorMap, Truncation,
};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
enum Label {
    Vacuum,
    One,
    Two,
    X,
    AliasX,
    Invalid,
}

#[test]
fn checked_generic_powi_zero_reuses_the_admitted_space_without_provider_work() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, ij| f64::from(ij == [0, 1]))
            .unwrap();
    reset_provider_queries(&provider);
    let identity = source.powi(0).unwrap();
    assert_no_provider_queries(&provider);
    assert!(std::ptr::eq(identity.provider(), provider.as_ref()));
    assert!(identity.runtime().shares_state_with(source.runtime()));
    assert_eq!(identity.codomain(), source.codomain());
    assert_eq!(identity.domain(), source.domain());
    assert_eq!(identity.block_count(), source.block_count());
    assert_eq!(identity.data(), &[1.0, 0.0, 0.0, 1.0]);
    for index in 0..source.block_count() {
        assert_eq!(identity.block(index).unwrap(), source.block(index).unwrap());
        assert_eq!(
            identity.block_fusion_trees(index).unwrap(),
            source.block_fusion_trees(index).unwrap()
        );
    }
}

#[test]
fn checked_generic_powi_matches_explicit_real_and_complex_oracles() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let real: TensorMap<_, f64> = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, ij| {
        [[2.0, 1.0], [3.0, 4.0]][ij[0]][ij[1]]
    })
    .unwrap();
    let complex = real.to_c64().scale(Complex64::new(1.0, 1.0));
    let real_oracles: &[(i32, &[f64])] = &[
        (0, &[1.0, 0.0, 0.0, 1.0]),
        (1, &[2.0, 3.0, 1.0, 4.0]),
        (2, &[7.0, 18.0, 6.0, 19.0]),
        (3, &[32.0, 93.0, 31.0, 94.0]),
        (-1, &[0.8, -0.6, -0.2, 0.4]),
        (-2, &[0.76, -0.72, -0.24, 0.28]),
    ];
    let complex_oracles: &[(i32, &[Complex64])] = &[
        (
            0,
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        ),
        (
            1,
            &[
                Complex64::new(2.0, 2.0),
                Complex64::new(3.0, 3.0),
                Complex64::new(1.0, 1.0),
                Complex64::new(4.0, 4.0),
            ],
        ),
        (
            2,
            &[
                Complex64::new(0.0, 14.0),
                Complex64::new(0.0, 36.0),
                Complex64::new(0.0, 12.0),
                Complex64::new(0.0, 38.0),
            ],
        ),
        (
            3,
            &[
                Complex64::new(-64.0, 64.0),
                Complex64::new(-186.0, 186.0),
                Complex64::new(-62.0, 62.0),
                Complex64::new(-188.0, 188.0),
            ],
        ),
        (
            -1,
            &[
                Complex64::new(0.4, -0.4),
                Complex64::new(-0.3, 0.3),
                Complex64::new(-0.1, 0.1),
                Complex64::new(0.2, -0.2),
            ],
        ),
        (
            -2,
            &[
                Complex64::new(0.0, -0.38),
                Complex64::new(0.0, 0.36),
                Complex64::new(0.0, 0.12),
                Complex64::new(0.0, -0.14),
            ],
        ),
    ];
    for &(exponent, expected) in real_oracles {
        assert!(real
            .powi(exponent)
            .unwrap()
            .data()
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (actual - expected).abs() < 1e-12));
    }
    for &(exponent, expected) in complex_oracles {
        assert!(complex
            .powi(exponent)
            .unwrap()
            .data()
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (*actual - *expected).norm() < 1e-12));
    }
}

#[test]
fn checked_generic_powi_rejects_nonendomorphisms_before_provider_work() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let wide = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let narrow = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&wide], [&narrow], |_, _| 1.0).unwrap();
    let before = source
        .data()
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
    for exponent in [0, 1, -1] {
        reset_provider_queries(&provider);
        match source.powi(exponent) {
            Err(GenericTensorError::Facade(tenet::typed::Error::InvalidArgument(message))) => {
                assert!(message.contains("endomorphism"));
            }
            other => panic!("unexpected powi({exponent}) result: {other:?}"),
        }
        assert_no_provider_queries(&provider);
        assert_eq!(
            source
                .data()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            before
        );
    }
}

#[test]
fn checked_generic_powi_singular_negative_powers_do_not_publish() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, ij| f64::from(ij == [0, 0]))
            .unwrap();
    assert_eq!(source.powi(2).unwrap().data(), &[1.0, 0.0, 0.0, 0.0]);
    let before = source
        .data()
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
    for exponent in [-1, -2] {
        assert!(matches!(
            source.powi(exponent),
            Err(GenericTensorError::Facade(tenet::typed::Error::Operation(
                _
            )))
        ));
        assert_eq!(
            source
                .data()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            before
        );
    }
}

#[test]
fn checked_generic_powi_i32_min_on_identity_is_exact() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let identity: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, ij| f64::from(ij[0] == ij[1]))
            .unwrap();
    assert_eq!(identity.powi(i32::MIN).unwrap().data(), identity.data());
}

#[cfg(feature = "racah-generated")]
fn assert_sun_checked_generic_powi_outer_multiplicity<D>(n: usize, adjoint: Vec<i64>)
where
    D: tenet::typed::TensorScalar + fmt::Debug + PartialEq,
{
    use tenet::typed::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(n).unwrap());
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(adjoint.clone(), 2)]).unwrap();
    let source: TensorMap<_, D> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, ij| {
            let row = ij[0] + 2 * ij[1];
            let col = ij[2] + 2 * ij[3];
            D::from_real(
                if trees.codomain_vertices() == trees.domain_vertices() && row == col {
                    2.0
                } else if trees.codomain_vertices()[0].get() == 1
                    && trees.domain_vertices()[0].get() == 2
                    && row == col
                {
                    1.0
                } else {
                    0.0
                },
            )
        })
        .unwrap();
    assert!((0..source.block_count()).any(|index| {
        let trees = source.block_fusion_trees(index).unwrap();
        trees.codomain_vertices()[0].get() == 2 || trees.domain_vertices()[0].get() == 2
    }));
    assert!((0..source.block_count()).any(|index| {
        let trees = source.block_fusion_trees(index).unwrap();
        trees.codomain_vertices()[0].get() == 1
            && trees.domain_vertices()[0].get() == 2
            && source.data()[source.block(index).unwrap().offset()] == D::from_real(1.0)
    }));

    let identity = source.powi(0).unwrap();
    assert!(std::ptr::eq(identity.provider(), provider.as_ref()));
    assert!(identity.runtime().shares_state_with(source.runtime()));
    assert_eq!(identity.codomain(), source.codomain());
    assert_eq!(identity.domain(), source.domain());
    for index in 0..source.block_count() {
        assert_eq!(identity.block(index).unwrap(), source.block(index).unwrap());
        assert_eq!(
            identity.block_fusion_trees(index).unwrap(),
            source.block_fusion_trees(index).unwrap()
        );
    }
    let squared = source.powi(2).unwrap();
    assert_eq!(squared.data(), source.compose(&source).unwrap().data());
    let inverse = source.powi(-1).unwrap();
    for product in [
        source.compose(&inverse).unwrap(),
        inverse.compose(&source).unwrap(),
    ] {
        assert_eq!(product.data(), identity.data());
    }
}

#[cfg(feature = "racah-generated")]
#[test]
fn checked_generic_powi_sun_outer_multiplicity_preserves_full_layout_and_power_laws() {
    for (n, adjoint) in [(3, vec![1, 1]), (4, vec![1, 0, 1])] {
        assert_sun_checked_generic_powi_outer_multiplicity::<f64>(n, adjoint.clone());
        assert_sun_checked_generic_powi_outer_multiplicity::<Complex64>(n, adjoint);
    }
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_endomorphism_row_and_column_tree_stacking_is_identical() {
    use std::collections::BTreeMap;

    use tenet::core::MultiplicityIndex;
    use tenet::typed::SUNFusionRule;

    type TreePlacement = (
        Vec<Vec<i64>>,
        Vec<Vec<i64>>,
        Vec<MultiplicityIndex>,
        usize,
        Vec<usize>,
    );

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    for (n, label) in [(3, vec![1, 1]), (4, vec![1, 0, 1])] {
        let provider = Arc::new(SUNFusionRule::new(n).unwrap());
        let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(label, 2)]).unwrap();
        let source: TensorMap<_, f64> =
            TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg, &leg], |_, _| 0.0).unwrap();
        let mut stacks = BTreeMap::<Vec<i64>, (Vec<TreePlacement>, Vec<TreePlacement>)>::new();
        for index in 0..source.block_count() {
            let block = source.block(index).unwrap();
            let trees = source.block_fusion_trees(index).unwrap();
            let entry = stacks.entry(trees.coupled().clone()).or_default();
            let row_key = (
                trees.codomain_uncoupled().to_vec(),
                trees.codomain_innerlines().to_vec(),
                trees.codomain_vertices().to_vec(),
            );
            if !entry.0.iter().any(|placed| {
                placed.0 == row_key.0 && placed.1 == row_key.1 && placed.2 == row_key.2
            }) {
                let offset = entry
                    .0
                    .iter()
                    .map(|placed| placed.4.iter().product::<usize>())
                    .sum();
                entry.0.push((
                    row_key.0,
                    row_key.1,
                    row_key.2,
                    offset,
                    block.shape()[..2].to_vec(),
                ));
            }
            let col_key = (
                trees.domain_uncoupled().to_vec(),
                trees.domain_innerlines().to_vec(),
                trees.domain_vertices().to_vec(),
            );
            if !entry.1.iter().any(|placed| {
                placed.0 == col_key.0 && placed.1 == col_key.1 && placed.2 == col_key.2
            }) {
                let offset = entry
                    .1
                    .iter()
                    .map(|placed| placed.4.iter().product::<usize>())
                    .sum();
                entry.1.push((
                    col_key.0,
                    col_key.1,
                    col_key.2,
                    offset,
                    block.shape()[2..].to_vec(),
                ));
            }
        }
        assert!(stacks.values().any(|(rows, _)| {
            rows.iter()
                .any(|row| row.2.iter().any(|vertex| vertex.get() > 1))
        }));
        for (sector, (rows, columns)) in stacks {
            assert_eq!(rows, columns, "SU({n}) stacking mismatch in {sector:?}");
        }
    }
}

#[cfg(feature = "racah-generated")]
fn assert_sun_checked_generic_eigh<D>(
    n: usize,
    label: Vec<i64>,
    off_diagonal: D,
    adjoint: impl Fn(D) -> D + Copy,
    real: impl Fn(D) -> f64 + Copy,
    norm_squared: f64,
    close: impl Fn(D, D) -> f64 + Copy,
) where
    D: tenet::typed::TensorScalar + fmt::Debug,
{
    use std::cell::Cell;

    use tenet::typed::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(n).unwrap());
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(label.clone(), 2)]).unwrap();
    let cross_sector = label.clone();
    let source: TensorMap<_, D> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, index| {
            let row = index[0] + 2 * index[1];
            let column = index[2] + 2 * index[3];
            if row != column {
                return D::from_real(0.0);
            }
            if trees.coupled() == &cross_sector {
                let mu_row = trees.codomain_vertices()[0].get();
                let mu_column = trees.domain_vertices()[0].get();
                match (mu_row, mu_column) {
                    (1, 1) => D::from_real(4.0 + 10.0 * row as f64),
                    (2, 2) => D::from_real(-1.0 + 10.0 * row as f64),
                    (1, 2) => off_diagonal,
                    (2, 1) => adjoint(off_diagonal),
                    _ => D::from_real(0.0),
                }
            } else if trees.codomain_vertices() == trees.domain_vertices() {
                D::from_real(50.0 + 10.0 * row as f64)
            } else {
                D::from_real(0.0)
            }
        })
        .unwrap();
    assert!((0..source.block_count()).any(|index| {
        let trees = source.block_fusion_trees(index).unwrap();
        trees.coupled() == &label
            && trees.codomain_vertices()[0].get() == 1
            && trees.domain_vertices()[0].get() == 2
            && source.block(index).unwrap().shape() == [2, 2, 2, 2]
            && source.data()[source.block(index).unwrap().offset()] == off_diagonal
    }));

    let (d, v) = source.eigh_full().unwrap();
    assert!(std::ptr::eq(d.provider(), provider.as_ref()));
    assert!(std::ptr::eq(v.provider(), provider.as_ref()));
    let dense_len = d.data().len();
    assert!(!format!("{d:?}").contains(&format!("elements: {dense_len}")));
    assert_eq!(v.codomain(), source.codomain());
    assert_eq!(v.domain(), d.codomain());
    for (actual, expected) in source
        .compose(&v)
        .unwrap()
        .data()
        .iter()
        .zip(v.compose(&d).unwrap().data())
    {
        assert!(close(*actual, *expected) < 1e-8);
    }

    let lazy_vh = v.adjoint().unwrap();
    let logical_vh = lazy_vh.data().to_vec();
    let position = Cell::new(0usize);
    let codomain = v.domain();
    let domain = v.codomain();
    let vh: TensorMap<_, D> =
        TensorMap::from_block_fn(&runtime, codomain.iter(), domain.iter(), |_, _| {
            let index = position.get();
            position.set(index + 1);
            logical_vh[index]
        })
        .unwrap();
    assert_same_checked_generic_layout_and_close(
        &v.compose(&d).unwrap().compose(&vh).unwrap(),
        &source,
        close,
    );

    let root = (6.25 + norm_squared).sqrt();
    let lambda_plus = 1.5 + root;
    let lambda_minus = 1.5 - root;
    let identity_codomain = d.codomain();
    let identity_domain = d.domain();
    let identity: TensorMap<_, D> = TensorMap::from_block_fn(
        &runtime,
        identity_codomain.iter(),
        identity_domain.iter(),
        |_, index| D::from_real(f64::from(index[0] == index[1])),
    )
    .unwrap();
    let mut selector = identity.clone();
    let mut skipped_target = false;
    for index in 0..d.block_count() {
        let block = d.block(index).unwrap();
        for diagonal in 0..block.shape()[0] {
            let value = d.data()
                [block.offset() + diagonal * block.strides()[0] + diagonal * block.strides()[1]];
            let scalar = real(value);
            if (scalar - lambda_plus).abs() < 1e-8 {
                assert!(!skipped_target, "target eigenvalue must be nondegenerate");
                skipped_target = true;
                continue;
            }
            let shifted = d
                .add(&identity, D::from_real(1.0), D::from_real(-scalar))
                .unwrap()
                .scale(D::from_real(1.0 / (lambda_plus - scalar)));
            selector = selector.compose(&shifted).unwrap();
        }
    }
    assert!(skipped_target);
    let projector = v.compose(&selector).unwrap().compose(&vh).unwrap();
    let expected: TensorMap<_, D> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, index| {
            let row = index[0] + 2 * index[1];
            let column = index[2] + 2 * index[3];
            if trees.coupled() != &label || row != 0 || column != 0 {
                return D::from_real(0.0);
            }
            let mu_row = trees.codomain_vertices()[0].get();
            let mu_column = trees.domain_vertices()[0].get();
            let inverse_denominator = D::from_real(1.0 / (lambda_plus - lambda_minus));
            match (mu_row, mu_column) {
                (1, 1) => D::from_real(4.0 - lambda_minus) * inverse_denominator,
                (2, 2) => D::from_real(-1.0 - lambda_minus) * inverse_denominator,
                (1, 2) => off_diagonal * inverse_denominator,
                (2, 1) => adjoint(off_diagonal) * inverse_denominator,
                _ => D::from_real(0.0),
            }
        })
        .unwrap();
    assert_same_checked_generic_layout_and_close(&projector, &expected, close);
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_checked_generic_eigh_cross_mu_projectors_for_both_dtypes() {
    for (n, label) in [(3, vec![1, 1]), (4, vec![1, 0, 1])] {
        assert_sun_checked_generic_eigh(
            n,
            label.clone(),
            1.0,
            |value| value,
            |value| value,
            1.0,
            |actual, expected| (actual - expected).abs(),
        );
        assert_sun_checked_generic_eigh(
            n,
            label,
            Complex64::new(1.0, 1.0),
            |value| value.conj(),
            |value| value.re,
            2.0,
            |actual, expected| (actual - expected).norm(),
        );
    }
}

#[cfg(feature = "racah-generated")]
trait SunEigInput:
    tenet::typed::TensorScalar + tenet_matrixalgebra::FactorScalar<Eig = Complex64> + fmt::Debug
{
    fn to_complex(
        source: &TensorMap<tenet::typed::SUNFusionRule, Self>,
    ) -> TensorMap<tenet::typed::SUNFusionRule, Complex64>;
}

#[cfg(feature = "racah-generated")]
impl SunEigInput for f64 {
    fn to_complex(
        source: &TensorMap<tenet::typed::SUNFusionRule, Self>,
    ) -> TensorMap<tenet::typed::SUNFusionRule, Complex64> {
        source.to_c64()
    }
}

#[cfg(feature = "racah-generated")]
impl SunEigInput for Complex64 {
    fn to_complex(
        source: &TensorMap<tenet::typed::SUNFusionRule, Self>,
    ) -> TensorMap<tenet::typed::SUNFusionRule, Complex64> {
        source.clone()
    }
}

#[cfg(feature = "racah-generated")]
fn assert_sun_checked_generic_eig<D>(n: usize, label: Vec<i64>)
where
    D: SunEigInput,
{
    use tenet::typed::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(n).unwrap());
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(label.clone(), 2)]).unwrap();
    let source: TensorMap<_, D> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, index| {
            let row = index[0] + 2 * index[1];
            let column = index[2] + 2 * index[3];
            if row != column {
                return D::from_real(0.0);
            }
            if trees.coupled() == &label {
                let mu_row = trees.codomain_vertices()[0].get();
                let mu_column = trees.domain_vertices()[0].get();
                let shift = 10.0 * row as f64;
                D::from_real(match (mu_row, mu_column) {
                    (1, 1) | (2, 2) => 1.0 + shift,
                    (1, 2) => -3.0,
                    (2, 1) => 1.0,
                    _ => 0.0,
                })
            } else if trees.codomain_vertices() == trees.domain_vertices() {
                D::from_real(50.0 + 10.0 * row as f64)
            } else {
                D::from_real(0.0)
            }
        })
        .unwrap();
    let (d, v) = source.eig_full().unwrap();
    assert!(std::ptr::eq(d.provider(), provider.as_ref()));
    assert!(std::ptr::eq(v.provider(), provider.as_ref()));
    let dense_len = d.data().len();
    assert!(!format!("{d:?}").contains(&format!("elements: {dense_len}")));
    let complex_source = D::to_complex(&source);
    let av = complex_source.compose(&v).unwrap();
    let vd = v.compose(&d).unwrap();
    assert!(av
        .data()
        .iter()
        .zip(vd.data())
        .all(|(actual, expected)| (*actual - *expected).norm() < 1.0e-8));
    let rebuilt = v.compose(&d).unwrap().compose(&v.inv().unwrap()).unwrap();
    assert_same_checked_generic_layout_and_close(&rebuilt, &complex_source, |actual, expected| {
        (actual - expected).norm()
    });

    let lambda_plus = Complex64::new(1.0, 3.0_f64.sqrt());
    let identity: TensorMap<_, Complex64> = TensorMap::from_block_fn(
        &runtime,
        d.codomain().iter(),
        d.domain().iter(),
        |_, index| Complex64::new(f64::from(index[0] == index[1]), 0.0),
    )
    .unwrap();
    let mut selector = identity.clone();
    let mut found_target = false;
    for index in 0..d.block_count() {
        let block = d.block(index).unwrap();
        for diagonal in 0..block.shape()[0] {
            let value = d.data()
                [block.offset() + diagonal * block.strides()[0] + diagonal * block.strides()[1]];
            if (value - lambda_plus).norm() < 1.0e-8 {
                assert!(!found_target);
                found_target = true;
            } else {
                selector = selector
                    .compose(
                        &d.add(&identity, Complex64::new(1.0, 0.0), -value)
                            .unwrap()
                            .scale(Complex64::new(1.0, 0.0) / (lambda_plus - value)),
                    )
                    .unwrap();
            }
        }
    }
    assert!(found_target);
    let projector = v
        .compose(&selector)
        .unwrap()
        .compose(&v.inv().unwrap())
        .unwrap();
    let root = 3.0_f64.sqrt();
    let expected: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, index| {
            let row = index[0] + 2 * index[1];
            let column = index[2] + 2 * index[3];
            if trees.coupled() != &label || row != 0 || column != 0 {
                return Complex64::new(0.0, 0.0);
            }
            match (
                trees.codomain_vertices()[0].get(),
                trees.domain_vertices()[0].get(),
            ) {
                (1, 1) | (2, 2) => Complex64::new(0.5, 0.0),
                (1, 2) => Complex64::new(0.0, root / 2.0),
                (2, 1) => Complex64::new(0.0, -1.0 / (2.0 * root)),
                _ => Complex64::new(0.0, 0.0),
            }
        })
        .unwrap();
    assert_same_checked_generic_layout_and_close(&projector, &expected, |actual, expected| {
        (actual - expected).norm()
    });
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_checked_generic_eig_nonnormal_cross_mu_projectors_for_both_dtypes() {
    for (n, label) in [(3, vec![1, 1]), (4, vec![1, 0, 1])] {
        assert_sun_checked_generic_eig::<f64>(n, label.clone());
        assert_sun_checked_generic_eig::<Complex64>(n, label);
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ToyError {
    InvalidSector,
    Decode,
    Algebra,
}

impl fmt::Display for ToyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for ToyError {}

struct CheckedOnlyToy {
    identity_tag: u8,
    fail_algebra: AtomicBool,
    fail_dim: AtomicBool,
    fail_decode: AtomicBool,
    algebra_queries: AtomicUsize,
    coefficient_queries: AtomicUsize,
    f_queries: AtomicUsize,
    r_queries: AtomicUsize,
    malformed_f: AtomicBool,
    invalid_style: AtomicBool,
    use_product_probe: bool,
    fractional_dim: bool,
    fail_f_on_query: AtomicUsize,
    identity_queries: AtomicUsize,
    style_queries: AtomicUsize,
    commit_identity_seen: AtomicBool,
    committed: AtomicBool,
    commit_count: AtomicUsize,
    postcommit_queries: AtomicUsize,
    commit_after_queries: AtomicUsize,
    queries_since_reset: AtomicUsize,
}

impl CheckedOnlyToy {
    fn new(identity_tag: u8) -> Self {
        Self {
            identity_tag,
            fail_algebra: AtomicBool::new(false),
            fail_dim: AtomicBool::new(false),
            fail_decode: AtomicBool::new(false),
            algebra_queries: AtomicUsize::new(0),
            coefficient_queries: AtomicUsize::new(0),
            f_queries: AtomicUsize::new(0),
            r_queries: AtomicUsize::new(0),
            malformed_f: AtomicBool::new(false),
            invalid_style: AtomicBool::new(false),
            use_product_probe: false,
            fractional_dim: false,
            fail_f_on_query: AtomicUsize::new(0),
            identity_queries: AtomicUsize::new(0),
            style_queries: AtomicUsize::new(0),
            commit_identity_seen: AtomicBool::new(false),
            committed: AtomicBool::new(false),
            commit_count: AtomicUsize::new(0),
            postcommit_queries: AtomicUsize::new(0),
            commit_after_queries: AtomicUsize::new(0),
            queries_since_reset: AtomicUsize::new(0),
        }
    }

    fn new_product_probe(identity_tag: u8) -> Self {
        Self {
            use_product_probe: true,
            ..Self::new(identity_tag)
        }
    }

    fn new_space_probe(identity_tag: u8) -> Self {
        Self {
            use_product_probe: true,
            fractional_dim: true,
            ..Self::new(identity_tag)
        }
    }

    fn x(&self) -> SectorId {
        SectorId::new(3)
    }

    fn probe_fusion_channels(left: SectorId, right: SectorId) -> SectorVec {
        let ids: &[usize] = match (left.id(), right.id()) {
            (0, x) | (x, 0) => return [SectorId::new(x)].into_iter().collect(),
            (3, 3) | (3, 1) | (1, 3) => &[3],
            (1, 1) => &[1],
            _ => &[],
        };
        ids.iter().copied().map(SectorId::new).collect()
    }

    fn probe_nsymbol(left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        if (left.id(), right.id(), coupled.id()) == (3, 3, 3) {
            2
        } else {
            usize::from(Self::probe_fusion_channels(left, right).contains(&coupled))
        }
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        if self.use_product_probe {
            Self::probe_fusion_channels(left, right)
        } else {
            match (left.id(), right.id()) {
                (0, x) | (x, 0) => [SectorId::new(x)].into_iter().collect(),
                (3, 3) => [SectorId::new(0), SectorId::new(3)].into_iter().collect(),
                _ => SectorVec::new(),
            }
        }
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        if self.use_product_probe {
            Self::probe_nsymbol(left, right, coupled)
        } else if (left.id(), right.id(), coupled.id()) == (3, 3, 3) {
            2
        } else {
            usize::from(self.fusion_channels(left, right).contains(&coupled))
        }
    }

    fn reset_commit_spy(&self) {
        self.commit_identity_seen.store(false, Ordering::Relaxed);
        self.committed.store(false, Ordering::Relaxed);
        self.commit_count.store(0, Ordering::Relaxed);
        self.postcommit_queries.store(0, Ordering::Relaxed);
        self.commit_after_queries.store(0, Ordering::Relaxed);
        self.queries_since_reset.store(0, Ordering::Relaxed);
    }

    fn arm_commit_spy_after_queries(&self, query_count: usize) {
        self.reset_commit_spy();
        assert!(query_count > 0);
        self.commit_after_queries
            .store(query_count, Ordering::Relaxed);
    }

    fn record_query(&self) {
        let query = self.queries_since_reset.fetch_add(1, Ordering::Relaxed) + 1;
        let commit_after = self.commit_after_queries.load(Ordering::Relaxed);
        if commit_after != 0 && query == commit_after {
            if !self.committed.swap(true, Ordering::Relaxed) {
                self.commit_count.fetch_add(1, Ordering::Relaxed);
            }
        } else if self.committed.load(Ordering::Relaxed) {
            self.postcommit_queries.fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn reset_provider_queries(provider: &CheckedOnlyToy) {
    for counter in [
        &provider.algebra_queries,
        &provider.coefficient_queries,
        &provider.f_queries,
        &provider.r_queries,
        &provider.identity_queries,
        &provider.style_queries,
        &provider.queries_since_reset,
    ] {
        counter.store(0, Ordering::Relaxed);
    }
}

fn assert_no_provider_queries(provider: &CheckedOnlyToy) {
    for counter in [
        &provider.algebra_queries,
        &provider.coefficient_queries,
        &provider.f_queries,
        &provider.r_queries,
        &provider.identity_queries,
        &provider.style_queries,
    ] {
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }
    assert_eq!(provider.queries_since_reset.load(Ordering::Relaxed), 0);
}

impl CheckedGenericFusion for CheckedOnlyToy {
    type Error = ToyError;

    fn rule_identity(&self) -> RuleIdentity {
        self.identity_queries.fetch_add(1, Ordering::Relaxed);
        self.record_query();
        if self.commit_after_queries.load(Ordering::Relaxed) == 0
            && self.f_queries.load(Ordering::Relaxed) > 0
        {
            self.commit_identity_seen.store(true, Ordering::Relaxed);
        }
        RuleIdentity::from_canonical_bytes::<Self>(
            0x677,
            Arc::<[u8]>::from([
                self.identity_tag,
                u8::from(self.use_product_probe),
                u8::from(self.fractional_dim),
            ]),
        )
    }

    fn fusion_style(&self) -> FusionStyleKind {
        self.style_queries.fetch_add(1, Ordering::Relaxed);
        self.record_query();
        if self.commit_after_queries.load(Ordering::Relaxed) == 0
            && self.commit_identity_seen.load(Ordering::Relaxed)
            && !self.committed.swap(true, Ordering::Relaxed)
        {
            self.commit_count.fetch_add(1, Ordering::Relaxed);
        }
        if self.invalid_style.load(Ordering::Relaxed) {
            FusionStyleKind::Unique
        } else {
            FusionStyleKind::Generic
        }
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        self.record_query();
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        self.record_query();
        SectorId::new(0)
    }

    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        self.record_query();
        self.algebra_queries.fetch_add(1, Ordering::Relaxed);
        Ok(sector)
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.record_query();
        self.algebra_queries.fetch_add(1, Ordering::Relaxed);
        if self.fail_algebra.load(Ordering::Relaxed) {
            return Err(ToyError::Algebra);
        }
        Ok(self.fusion_channels(left, right))
    }

    fn try_fusion_channels_in_table(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.record_query();
        self.algebra_queries.fetch_add(1, Ordering::Relaxed);
        Ok(self.fusion_channels(left, right))
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, Self::Error> {
        self.record_query();
        self.algebra_queries.fetch_add(1, Ordering::Relaxed);
        Ok(self.nsymbol(left, right, coupled))
    }
}

impl CheckedGenericRigidSymbols for CheckedOnlyToy {
    type Scalar = f64;

    fn try_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.record_query();
        self.coefficient_queries.fetch_add(1, Ordering::Relaxed);
        if self.fail_algebra.load(Ordering::Relaxed) || self.fail_dim.load(Ordering::Relaxed) {
            return Err(ToyError::Algebra);
        }
        Ok(if sector.id() == 3 {
            if self.fractional_dim {
                2.5_f64
            } else {
                1.0 + 2.0_f64.sqrt()
            }
            .sqrt()
        } else {
            1.0
        })
    }

    fn try_inv_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.record_query();
        self.coefficient_queries.fetch_add(1, Ordering::Relaxed);
        Ok(if sector.id() == 3 {
            1.0 / if self.fractional_dim {
                2.5_f64
            } else {
                1.0 + 2.0_f64.sqrt()
            }
            .sqrt()
        } else {
            1.0
        })
    }

    fn try_frobenius_schur_phase_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.record_query();
        self.coefficient_queries.fetch_add(1, Ordering::Relaxed);
        let _ = sector;
        Ok(1.0)
    }

    fn try_f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Result<GenericFArray<f64>, Self::Error> {
        self.record_query();
        self.coefficient_queries.fetch_add(1, Ordering::Relaxed);
        let query = self.f_queries.fetch_add(1, Ordering::Relaxed) + 1;
        if self.fail_algebra.load(Ordering::Relaxed)
            || self.fail_f_on_query.load(Ordering::Relaxed) == query
        {
            return Err(ToyError::Algebra);
        }
        let shape = (
            self.nsymbol(a, b, e),
            self.nsymbol(e, c, d),
            self.nsymbol(b, c, f),
            self.nsymbol(a, f, d),
        );
        let len = shape.0 * shape.1 * shape.2 * shape.3;
        let symbol = if self.use_product_probe {
            let data = (0..len)
                .map(|index| {
                    let magnitude = (index + 1) as f64;
                    if index % 2 == 0 {
                        magnitude
                    } else {
                        -magnitude
                    }
                })
                .collect();
            GenericFArray::new(data, shape)
        } else if e == f {
            let cols = shape.2 * shape.3;
            GenericFArray::new(
                (0..len)
                    .map(|index| f64::from(index / cols == index % cols))
                    .collect(),
                shape,
            )
        } else {
            GenericFArray::new(vec![0.0; len], shape)
        };
        if self.malformed_f.load(Ordering::Relaxed) {
            Ok(GenericFArray::new(
                symbol.data().to_vec(),
                (1, 1, symbol.data().len(), 1),
            ))
        } else {
            Ok(symbol)
        }
    }

    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<f64>, Self::Error> {
        self.record_query();
        self.coefficient_queries.fetch_add(1, Ordering::Relaxed);
        self.r_queries.fetch_add(1, Ordering::Relaxed);
        if self.fail_algebra.load(Ordering::Relaxed) {
            return Err(ToyError::Algebra);
        }
        let rows = self.nsymbol(a, b, c);
        Ok(GenericRMatrix::new(
            (0..rows * rows)
                .map(|index| f64::from(index / rows == index % rows))
                .collect(),
            rows,
            rows,
        ))
    }
}

impl TypedSectorAdmission for CheckedOnlyToy {
    type Sector = Label;
    type Error = ToyError;
    type Mode = CheckedGenericAdmissionMode;

    fn typed_rule_identity(&self) -> RuleIdentity {
        CheckedGenericFusion::rule_identity(self)
    }

    fn try_encode_label(&self, sector: &Self::Sector) -> Result<SectorId, Self::Error> {
        self.record_query();
        match sector {
            Label::Vacuum => Ok(self.vacuum()),
            Label::One if self.use_product_probe => Ok(SectorId::new(1)),
            Label::Two if self.use_product_probe => Ok(SectorId::new(2)),
            Label::X => Ok(self.x()),
            Label::AliasX => Ok(self.x()),
            Label::One | Label::Two | Label::Invalid => Err(ToyError::InvalidSector),
        }
    }

    fn try_decode_label(&self, sector: SectorId) -> Result<Self::Sector, Self::Error> {
        self.record_query();
        if self.fail_decode.load(Ordering::Relaxed) {
            return Err(ToyError::Decode);
        }
        if sector == self.vacuum() {
            Ok(Label::Vacuum)
        } else if self.use_product_probe && sector == SectorId::new(1) {
            Ok(Label::One)
        } else if self.use_product_probe && sector == SectorId::new(2) {
            Ok(Label::Two)
        } else if sector == self.x() {
            Ok(Label::X)
        } else {
            Err(ToyError::InvalidSector)
        }
    }

    fn try_dual_id(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        CheckedGenericFusion::try_dual(self, sector)
    }
}

#[test]
fn checked_generic_diagonal_is_compact_canonical_and_provider_owned() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let bond =
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 2), (Label::Vacuum, 1)])
            .unwrap()
            .try_dual()
            .unwrap();
    let real = TensorMap::<_, f64>::diagonal(
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: Label::X,
                values: vec![2.0, 3.0],
            },
            SectorSpectrum {
                sector: Label::Vacuum,
                values: vec![1.0],
            },
        ],
    )
    .unwrap();
    assert!(std::ptr::eq(real.provider(), provider.as_ref()));
    assert_eq!(real.codomain()[0], bond);
    assert_eq!(real.domain()[0], bond);
    assert!(!format!("{real:?}").contains("elements: 5"));
    assert_eq!(
        real.diagonal_spectrum().unwrap().unwrap(),
        [
            SectorSpectrum {
                sector: Label::Vacuum,
                values: vec![1.0],
            },
            SectorSpectrum {
                sector: Label::X,
                values: vec![2.0, 3.0],
            },
        ]
    );
    let adjoint = real.adjoint().unwrap();
    assert!(std::ptr::eq(adjoint.provider(), provider.as_ref()));
    assert!(adjoint.network_reuse_class(false) == NetworkReuseClass::LazyAdjoint);
    assert_eq!(real.data(), &[1.0, 2.0, 0.0, 0.0, 3.0]);
    assert_eq!(adjoint.data(), real.data());
    assert_eq!(
        adjoint
            .adjoint()
            .unwrap()
            .diagonal_spectrum()
            .unwrap()
            .unwrap(),
        real.diagonal_spectrum().unwrap().unwrap()
    );

    let complex = TensorMap::<_, Complex64>::diagonal(
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: Label::Vacuum,
                values: vec![Complex64::new(1.0, 1.0)],
            },
            SectorSpectrum {
                sector: Label::X,
                values: vec![Complex64::new(2.0, -1.0), Complex64::new(3.0, 2.0)],
            },
        ],
    )
    .unwrap();
    assert!(std::ptr::eq(complex.provider(), provider.as_ref()));
    assert_eq!(
        complex.adjoint().unwrap().data()[0],
        Complex64::new(1.0, -1.0)
    );
}

#[test]
fn checked_generic_diagonal_rejects_before_layout_and_preserves_error_precedence() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let bond =
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::Vacuum, 1), (Label::X, 2)])
            .unwrap();

    let compact = TensorMap::<_, f64>::diagonal(
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: Label::Vacuum,
                values: vec![1.0],
            },
            SectorSpectrum {
                sector: Label::X,
                values: vec![2.0, 3.0],
            },
        ],
    )
    .unwrap();
    provider.fail_decode.store(true, Ordering::Relaxed);
    assert!(matches!(
        compact.diagonal_spectrum(),
        Err(GenericTensorError::Structure(
            CheckedGenericStructureError::Provider(ToyError::Decode)
        ))
    ));
    provider.fail_decode.store(false, Ordering::Relaxed);

    reset_provider_queries(&provider);
    assert!(matches!(
        TensorMap::<_, f64>::diagonal(
            &runtime,
            &bond,
            [
                SectorSpectrum { sector: Label::Invalid, values: vec![1.0] },
                SectorSpectrum { sector: Label::Invalid, values: vec![2.0] },
            ],
        ),
        Err(GenericTensorError::Facade(tenet::typed::Error::InvalidArgument(message)))
            if message.contains("more than once")
    ));
    assert_no_provider_queries(&provider);

    assert!(matches!(
        TensorMap::<_, f64>::diagonal(
            &runtime,
            &bond,
            [
                SectorSpectrum {
                    sector: Label::X,
                    values: vec![1.0, 2.0]
                },
                SectorSpectrum {
                    sector: Label::Invalid,
                    values: vec![3.0]
                },
            ],
        ),
        Err(GenericTensorError::Structure(
            CheckedGenericStructureError::Provider(ToyError::InvalidSector)
        ))
    ));
    assert!(matches!(
        TensorMap::<_, f64>::diagonal(
            &runtime,
            &bond,
            [
                SectorSpectrum { sector: Label::X, values: vec![1.0, 2.0] },
                SectorSpectrum { sector: Label::AliasX, values: vec![3.0, 4.0] },
            ],
        ),
        Err(GenericTensorError::Facade(tenet::typed::Error::InvalidArgument(message)))
            if message.contains("both encode")
    ));

    provider.invalid_style.store(true, Ordering::Relaxed);
    for (spectra, expected) in [
        (
            vec![SectorSpectrum {
                sector: Label::Vacuum,
                values: vec![1.0],
            }],
            "missing",
        ),
        (
            vec![
                SectorSpectrum {
                    sector: Label::Vacuum,
                    values: vec![1.0],
                },
                SectorSpectrum {
                    sector: Label::One,
                    values: vec![2.0],
                },
            ],
            "missing",
        ),
        (
            vec![
                SectorSpectrum {
                    sector: Label::Vacuum,
                    values: vec![1.0],
                },
                SectorSpectrum {
                    sector: Label::X,
                    values: vec![2.0],
                },
                SectorSpectrum {
                    sector: Label::One,
                    values: vec![4.0],
                },
            ],
            "unknown",
        ),
        (
            vec![
                SectorSpectrum {
                    sector: Label::Vacuum,
                    values: vec![1.0],
                },
                SectorSpectrum {
                    sector: Label::X,
                    values: vec![2.0],
                },
            ],
            "length",
        ),
    ] {
        assert!(matches!(
            TensorMap::<_, f64>::diagonal(&runtime, &bond, spectra),
            Err(GenericTensorError::Facade(tenet::typed::Error::InvalidArgument(message)))
                if message.contains(expected)
        ));
    }

    assert!(matches!(
        TensorMap::<_, f64>::diagonal(
            &runtime,
            &bond,
            [
                SectorSpectrum {
                    sector: Label::Vacuum,
                    values: vec![1.0]
                },
                SectorSpectrum {
                    sector: Label::X,
                    values: vec![2.0, 3.0]
                },
            ],
        ),
        Err(GenericTensorError::Structure(_))
    ));
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_checked_generic_diagonal_constructs_standalone_compact_blocks() {
    use tenet::typed::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    for (n, adjoint) in [(3, vec![1, 1]), (4, vec![1, 0, 1])] {
        let provider = Arc::new(SUNFusionRule::new(n).unwrap());
        let adjoint_id = provider.encode_dynkin(&adjoint).unwrap();
        assert_eq!(
            provider
                .try_nsymbol(adjoint_id, adjoint_id, adjoint_id)
                .unwrap(),
            2
        );
        let bond =
            GradedSpace::try_new_with_arc(Arc::clone(&provider), [(adjoint.clone(), 2)]).unwrap();
        let diagonal = TensorMap::<_, f64>::diagonal(
            &runtime,
            &bond,
            [SectorSpectrum {
                sector: adjoint,
                values: vec![2.0, 3.0],
            }],
        )
        .unwrap();
        assert!(std::ptr::eq(diagonal.provider(), provider.as_ref()));
        assert!(!format!("{diagonal:?}").contains("elements: 4"));
        // This provider has outer multiplicity two, but a rank-one spectrum has no μ axis.
        assert_eq!(diagonal.block_count(), 1);
        assert_eq!(diagonal.block(0).unwrap().shape(), &[2, 2]);
        assert_eq!(diagonal.block(0).unwrap().strides(), &[1, 2]);
        assert_eq!(diagonal.data(), &[2.0, 0.0, 0.0, 3.0]);
    }
}

#[test]
fn checked_generic_space_algebra_keeps_multiplicity_dimensions_and_failures_typed() {
    let provider = Arc::new(CheckedOnlyToy::new_space_probe(7));
    let rhs_provider = Arc::new(CheckedOnlyToy::new_space_probe(7));
    let left = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let right = GradedSpace::try_new_with_arc(Arc::clone(&rhs_provider), [(Label::X, 3)]).unwrap();

    let dim = left.dim().unwrap();
    assert!((dim - 5.0).abs() < 1.0e-12);
    assert_ne!(dim, 6.0);
    let fused = left.fuse(&right).unwrap();
    assert_eq!(fused.degeneracy(&Label::X).unwrap(), 12);
    assert!(std::ptr::eq(fused.provider(), provider.as_ref()));
    assert!(!std::ptr::eq(fused.provider(), rhs_provider.as_ref()));
    let summed = left.oplus(&right).unwrap();
    assert_eq!(summed.degeneracy(&Label::X).unwrap(), 5);
    assert!(std::ptr::eq(summed.provider(), provider.as_ref()));
    assert!(!std::ptr::eq(summed.provider(), rhs_provider.as_ref()));
    let unit = left.unitspace().unwrap();
    assert!(std::ptr::eq(unit.provider(), provider.as_ref()));
    assert_eq!(unit.degeneracy(&Label::Vacuum).unwrap(), 1);

    let foreign_provider = Arc::new(CheckedOnlyToy::new_space_probe(8));
    let foreign =
        GradedSpace::try_new_with_arc(Arc::clone(&foreign_provider), [(Label::X, 1)]).unwrap();
    let before = provider.algebra_queries.load(Ordering::Relaxed)
        + foreign_provider.algebra_queries.load(Ordering::Relaxed);
    assert!(matches!(
        left.oplus(&foreign),
        Err(GenericTensorError::Facade(
            tenet::prelude::Error::RuleMismatch
        ))
    ));
    assert!(matches!(
        left.fuse(&foreign),
        Err(GenericTensorError::Facade(
            tenet::prelude::Error::RuleMismatch
        ))
    ));
    assert_eq!(
        provider.algebra_queries.load(Ordering::Relaxed)
            + foreign_provider.algebra_queries.load(Ordering::Relaxed),
        before
    );

    provider.fail_algebra.store(true, Ordering::Relaxed);
    assert!(matches!(
        left.fuse(&right),
        Err(GenericTensorError::Structure(
            CheckedGenericStructureError::Provider(ToyError::Algebra)
        ))
    ));
    assert_eq!(left.degeneracy(&Label::X).unwrap(), 2);
    assert_eq!(right.degeneracy(&Label::X).unwrap(), 3);
}

#[test]
fn checked_only_provider_uses_ordinary_typed_ownership_and_vertices() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let first = Arc::new(CheckedOnlyToy::new(0));
    let second = Arc::new(CheckedOnlyToy::new(0));
    let left = GradedSpace::try_new_with_arc(Arc::clone(&first), [(Label::X, 2)]).unwrap();
    let right = GradedSpace::try_new_with_arc(Arc::clone(&second), [(Label::X, 2)]).unwrap();

    let tensor: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&left, &right], [&right]).unwrap();
    assert!(std::ptr::eq(tensor.provider(), first.as_ref()));
    assert_eq!(tensor.rank(), 3);
    assert_eq!(tensor.block_count(), 2);
    let vertices: Vec<_> = (0..tensor.block_count())
        .map(|index| {
            let trees = tensor.block_fusion_trees(index).unwrap();
            assert_eq!(trees.coupled(), &Label::X);
            assert_eq!(trees.codomain_uncoupled(), &[Label::X, Label::X]);
            assert!(trees.codomain_innerlines().is_empty());
            assert_eq!(trees.domain_uncoupled(), &[Label::X]);
            assert!(trees.domain_vertices().is_empty());
            trees.codomain_vertices()[0].get()
        })
        .collect();
    assert_eq!(vertices, [1, 2]);
    assert_eq!(left.sectors().unwrap(), [Label::X]);

    let clone = tensor.clone();
    assert!(std::ptr::eq(clone.provider(), first.as_ref()));
    assert_eq!(clone.data().as_ptr(), tensor.data().as_ptr());
}

#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn checked_only_provider_roundtrips_through_typed_cuda_without_algebra_dispatch() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let codomain =
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::Vacuum, 1), (Label::X, 2)])
            .unwrap();
    let domain =
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::Vacuum, 1), (Label::X, 3)])
            .and_then(|space| space.try_dual())
            .unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&codomain], [&domain], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
    let vertex_leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let vertex_source: TensorMap<_, f64> = TensorMap::from_block_fn(
        &runtime,
        [&vertex_leg, &vertex_leg],
        [&vertex_leg],
        |trees, indices| {
            trees.codomain_vertices()[0].get() as f64 + indices.iter().sum::<usize>() as f64
        },
    )
    .unwrap();
    let block_structure = |tensor: &TensorMap<CheckedOnlyToy, f64>| {
        (0..tensor.block_count())
            .map(|index| {
                let block = tensor.block(index).unwrap();
                (
                    block.key().clone(),
                    tensor.block_fusion_trees(index).unwrap(),
                    block.offset(),
                    block.shape().to_vec(),
                    block.strides().to_vec(),
                )
            })
            .collect::<Vec<_>>()
    };
    let structure = |tensor: &TensorMap<CheckedOnlyToy, f64>| {
        let mut codomain_legs = Vec::new();
        let mut domain_legs = Vec::new();
        for index in 0..tensor.block_count() {
            let block = tensor.block(index).unwrap();
            let trees = tensor.block_fusion_trees(index).unwrap();
            let tenet::core::BlockKey::FusionTree(raw_trees) = block.key() else {
                panic!("checked Generic tensors use fusion-tree block keys")
            };
            assert_eq!(trees.codomain_uncoupled().len(), 1);
            assert_eq!(trees.domain_uncoupled().len(), 1);
            assert_eq!(block.shape().len(), 2);
            codomain_legs.push((
                trees.codomain_uncoupled()[0],
                block.shape()[0],
                raw_trees.codomain_tree().is_dual()[0],
            ));
            domain_legs.push((
                trees.domain_uncoupled()[0],
                block.shape()[1],
                raw_trees.domain_tree().is_dual()[0],
            ));
        }
        codomain_legs.sort_unstable();
        codomain_legs.dedup();
        domain_legs.sort_unstable();
        domain_legs.dedup();
        (codomain_legs, domain_legs, block_structure(tensor))
    };
    let expected_structure = structure(&source);
    let expected_vertex_structure = block_structure(&vertex_source);
    assert_eq!(
        expected_structure.0,
        [(Label::Vacuum, 1, false), (Label::X, 2, false)]
    );
    assert_eq!(
        expected_structure.1,
        [(Label::Vacuum, 1, true), (Label::X, 3, true)]
    );
    let expected = source.data().to_vec();
    let expected_vertex_data = vertex_source.data().to_vec();
    provider.algebra_queries.store(0, Ordering::Relaxed);
    provider.coefficient_queries.store(0, Ordering::Relaxed);

    let device = source.to_cuda().unwrap();
    let restored = device.to_host().unwrap();
    let vertex_device = vertex_source.to_cuda().unwrap();
    let vertex_restored = vertex_device.to_host().unwrap();

    assert!(std::ptr::eq(restored.provider(), provider.as_ref()));
    assert_eq!(restored.data(), expected);
    assert_eq!(structure(&restored), expected_structure);
    assert_eq!(vertex_restored.data(), expected_vertex_data);
    assert_eq!(block_structure(&vertex_restored), expected_vertex_structure);
    assert_eq!(provider.algebra_queries.load(Ordering::Relaxed), 0);
    assert_eq!(provider.coefficient_queries.load(Ordering::Relaxed), 0);
}

#[test]
fn checked_only_multiplicity_two_transforms_keep_the_source_authority() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg, &leg], [], |trees, _| {
            trees.codomain_vertices()[0].get() as f64
        })
        .unwrap();
    let snapshot = |tensor: &TensorMap<CheckedOnlyToy, f64>| {
        (0..tensor.block_count())
            .map(|index| tensor.block_fusion_trees(index).unwrap())
            .collect::<Vec<_>>()
    };
    let source_snapshot = snapshot(&source);
    provider.coefficient_queries.store(0, Ordering::Relaxed);
    let error = source.braid(&[1, 0, 2], &[], &[0, 1]).unwrap_err();
    assert!(matches!(error, GenericTensorError::Facade(_)));
    assert_eq!(provider.coefficient_queries.load(Ordering::Relaxed), 0);

    let permuted = source.permute(&[1, 0, 2], &[]).unwrap();
    assert!(std::ptr::eq(permuted.provider(), provider.as_ref()));
    let restored = permuted.permute(&[1, 0, 2], &[]).unwrap();
    assert_eq!(snapshot(&restored), source_snapshot);
    for (actual, expected) in restored.data().iter().zip(source.data()) {
        assert!((actual - expected).abs() <= 1e-12);
    }

    let braided = source.braid(&[1, 0, 2], &[], &[0, 1, 2]).unwrap();
    assert!(std::ptr::eq(braided.provider(), provider.as_ref()));

    provider.fail_algebra.store(true, Ordering::Relaxed);
    let error = source.permute(&[1, 0, 2], &[]).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Plan(tenet::typed::CheckedGenericPlanError::Provider(
            ToyError::Algebra
        ))
    ));
}

#[test]
fn checked_generic_reductions_cover_real_complex_dense_payloads() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            (indices.iter().sum::<usize>() + 1) as f64
        })
        .unwrap();
    let inner = source.inner(&source).unwrap();
    assert!(inner.is_finite());
    assert!((source.norm().unwrap() * source.norm().unwrap() - inner).abs() < 1e-12);
    assert!(source.tr().unwrap().is_finite());
    let complex = source.to_c64();
    assert!(complex.inner(&complex).unwrap().re.is_finite());
    assert!(complex.norm().unwrap().is_finite());
    assert!(complex.tr().unwrap().re.is_finite());
    assert!(provider.coefficient_queries.load(Ordering::Relaxed) > 0);
}

#[test]
fn checked_generic_host_add_scale_cover_real_and_complex_payloads() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            (indices.iter().sum::<usize>() + 1) as f64
        })
        .unwrap();
    let added = source.add(&source, 2.0, -1.0).unwrap();
    assert_eq!(added.data(), source.data());
    let scaled = source.scale(3.0);
    assert!(scaled
        .data()
        .iter()
        .zip(source.data())
        .all(|(a, b)| (*a - 3.0 * *b).abs() < 1e-12));

    let complex = source.to_c64();
    let added = complex
        .add(
            &complex,
            Complex64::new(2.0, 0.0),
            Complex64::new(-1.0, 0.0),
        )
        .unwrap();
    assert_eq!(added.data(), complex.data());
    let scaled = complex.scale(Complex64::new(0.5, -1.0));
    assert!(scaled
        .data()
        .iter()
        .zip(complex.data())
        .all(|(a, b)| (*a - *b * Complex64::new(0.5, -1.0)).norm() < 1e-12));
}

#[test]
fn checked_generic_add_rejects_runtime_before_layout_without_queries() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let foreign_runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let narrow = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let wide = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let left: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&narrow], [&narrow], |_, _| 1.0).unwrap();
    let right: TensorMap<_, f64> =
        TensorMap::from_block_fn(&foreign_runtime, [&wide], [&wide], |_, _| 2.0).unwrap();
    reset_provider_queries(&provider);
    let before = left.data().to_vec();
    let error = left.add(&right, 1.0, 1.0).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Facade(tenet::prelude::Error::RuntimeMismatch)
    ));
    assert_eq!(left.data(), before.as_slice());
    assert_eq!(provider.algebra_queries.load(Ordering::Relaxed), 0);
    assert_eq!(provider.coefficient_queries.load(Ordering::Relaxed), 0);
}

#[test]
fn checked_generic_add_rejects_layout_mismatch_without_queries() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let narrow = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let wide = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let left: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&narrow], [&narrow], |_, _| 1.0).unwrap();
    let right: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&wide], [&wide], |_, _| 2.0).unwrap();
    reset_provider_queries(&provider);
    let before = left.data().to_vec();
    let error = left.add(&right, 1.0, 1.0).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Facade(tenet::prelude::Error::InvalidArgument(_))
    ));
    assert_eq!(left.data(), before.as_slice());
    assert_eq!(provider.algebra_queries.load(Ordering::Relaxed), 0);
    assert_eq!(provider.coefficient_queries.load(Ordering::Relaxed), 0);
}

#[test]
fn checked_generic_add_assign_rejects_runtime_before_layout_and_preserves_receiver() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let foreign_runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let narrow = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let wide = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let mut left: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&narrow], [&narrow], |_, _| 1.0).unwrap();
    let right: TensorMap<_, f64> =
        TensorMap::from_block_fn(&foreign_runtime, [&wide], [&wide], |_, _| 2.0).unwrap();
    let before_data = left.data().to_vec();
    let before_trees = (0..left.block_count())
        .map(|index| left.block_fusion_trees(index).unwrap())
        .collect::<Vec<_>>();
    reset_provider_queries(&provider);
    let error = left.add_assign(&right, 1.0, 1.0).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Facade(tenet::prelude::Error::RuntimeMismatch)
    ));
    assert_eq!(left.data(), before_data.as_slice());
    assert_eq!(
        (0..left.block_count())
            .map(|index| left.block_fusion_trees(index).unwrap())
            .collect::<Vec<_>>(),
        before_trees
    );
    assert_eq!(provider.algebra_queries.load(Ordering::Relaxed), 0);
    assert_eq!(provider.coefficient_queries.load(Ordering::Relaxed), 0);
}

#[test]
fn checked_generic_add_assign_rejects_layout_mismatch_and_preserves_receiver() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let narrow = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let wide = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let mut left: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&narrow], [&narrow], |_, _| 1.0).unwrap();
    let right: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&wide], [&wide], |_, _| 2.0).unwrap();
    let before_data = left.data().to_vec();
    let before_trees = (0..left.block_count())
        .map(|index| left.block_fusion_trees(index).unwrap())
        .collect::<Vec<_>>();
    reset_provider_queries(&provider);
    let error = left.add_assign(&right, 1.0, 1.0).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Facade(tenet::prelude::Error::InvalidArgument(_))
    ));
    assert_eq!(left.data(), before_data.as_slice());
    assert_eq!(
        (0..left.block_count())
            .map(|index| left.block_fusion_trees(index).unwrap())
            .collect::<Vec<_>>(),
        before_trees
    );
    assert_eq!(provider.algebra_queries.load(Ordering::Relaxed), 0);
    assert_eq!(provider.coefficient_queries.load(Ordering::Relaxed), 0);
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_checked_generic_unit_insert_remove_preserves_authority_and_payload() {
    use tenet::prelude::GenericUnitTensorMapExt;
    use tenet::typed::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    for n in [3, 4] {
        let provider = Arc::new(SUNFusionRule::new(n).unwrap());
        let label = if n == 3 { vec![1, 1] } else { vec![1, 0, 1] };
        let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(label, 1)]).unwrap();
        let source: TensorMap<_, f64> =
            TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, _| 2.0).unwrap();

        assert!(source.remove_unit(0).is_err());
        let inserted = source.insert_left_unit(0, false).unwrap();
        assert!(std::ptr::eq(inserted.provider(), provider.as_ref()));
        assert_eq!(inserted.data().as_ptr(), source.data().as_ptr());
        let removed = inserted.remove_unit(0).unwrap();
        assert!(std::ptr::eq(removed.provider(), provider.as_ref()));
        assert_eq!(removed.data().as_ptr(), source.data().as_ptr());
        assert_eq!(removed.data(), source.data());
    }
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_checked_generic_compact_qr_preserves_provider_and_reconstructs() {
    use tenet::typed::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(vec![1, 1], 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |trees, _| {
            trees.coupled().iter().sum::<i64>() as f64 + 1.0
        })
        .unwrap();

    let (q, r) = source.qr_compact().unwrap();
    assert!(std::ptr::eq(q.provider(), provider.as_ref()));
    assert!(std::ptr::eq(r.provider(), provider.as_ref()));
    let rebuilt = q.compose(&r).unwrap();
    assert!(rebuilt
        .data()
        .iter()
        .zip(source.data())
        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-10));

    let complex = source.to_c64();
    let (complex_q, complex_r) = complex.qr_compact().unwrap();
    assert!(std::ptr::eq(complex_q.provider(), provider.as_ref()));
    assert!(std::ptr::eq(complex_r.provider(), provider.as_ref()));
    let complex_rebuilt = complex_q.compose(&complex_r).unwrap();
    assert!(complex_rebuilt
        .data()
        .iter()
        .zip(complex.data())
        .all(|(actual, expected)| (*actual - *expected).norm() < 1.0e-10));
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_checked_generic_compact_svd_preserves_provider_and_reconstructs() {
    use tenet::typed::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(vec![1, 1], 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |trees, _| {
            trees.coupled().iter().sum::<i64>() as f64 + 1.0
        })
        .unwrap();

    let (u, s, vh) = source.svd_compact().unwrap();
    assert!(std::ptr::eq(u.provider(), provider.as_ref()));
    assert!(std::ptr::eq(s.provider(), provider.as_ref()));
    assert!(std::ptr::eq(vh.provider(), provider.as_ref()));
    let rebuilt = u.compose(&s).unwrap().compose(&vh).unwrap();
    assert!(rebuilt
        .data()
        .iter()
        .zip(source.data())
        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-10));

    let complex = source.to_c64();
    let (complex_u, complex_s, complex_vh) = complex.svd_compact().unwrap();
    assert!(std::ptr::eq(complex_u.provider(), provider.as_ref()));
    assert!(std::ptr::eq(complex_s.provider(), provider.as_ref()));
    assert!(std::ptr::eq(complex_vh.provider(), provider.as_ref()));
    let complex_rebuilt = complex_u
        .compose(&complex_s)
        .unwrap()
        .compose(&complex_vh)
        .unwrap();
    assert!(complex_rebuilt
        .data()
        .iter()
        .zip(complex.data())
        .all(|(actual, expected)| (*actual - *expected).norm() < 1.0e-10));
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_checked_generic_dense_sqrt_preserves_svd_bond_and_principal_branch() {
    use tenet::typed::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    for n in [3, 4] {
        let provider = Arc::new(SUNFusionRule::new(n).unwrap());
        let label = if n == 3 { vec![1, 1] } else { vec![1, 0, 1] };
        let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(label, 1)]).unwrap();
        let source: TensorMap<_, f64> =
            TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |trees, _| {
                trees.codomain_vertices()[0].get() as f64 + 1.0
            })
            .unwrap();
        let (_, s, _) = source.svd_compact().unwrap();
        let root = s.sqrt().unwrap();
        assert!(std::ptr::eq(root.provider(), s.provider()));
        assert_eq!(root.codomain(), s.codomain());
        assert_eq!(root.domain(), s.domain());
        assert!(root.runtime().shares_state_with(s.runtime()));
        assert!(root
            .compose(&root)
            .unwrap()
            .data()
            .iter()
            .zip(s.data())
            .all(|(actual, expected)| (actual - expected).abs() < 1.0e-10));
        let complex = source.to_c64();
        let (_, s, _) = complex.svd_compact().unwrap();
        let root = s.sqrt().unwrap();
        assert!(std::ptr::eq(root.provider(), s.provider()));
        assert_eq!(root.codomain(), s.codomain());
        assert_eq!(root.domain(), s.domain());
        assert!(root.runtime().shares_state_with(s.runtime()));
        assert!(root
            .compose(&root)
            .unwrap()
            .data()
            .iter()
            .zip(s.data())
            .all(|(actual, expected)| (*actual - *expected).norm() < 1.0e-10));
        let negative: TensorMap<_, Complex64> =
            TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
                if indices[0] == indices[1] {
                    Complex64::new(-1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                }
            })
            .unwrap();
        let principal = negative.sqrt().unwrap();
        assert!(principal
            .data()
            .iter()
            .any(|value| *value == Complex64::new(0.0, 1.0)));
        assert!(principal.data().iter().all(|value| {
            *value == Complex64::new(0.0, 0.0) || *value == Complex64::new(0.0, 1.0)
        }));
        assert!(principal
            .compose(&principal)
            .unwrap()
            .data()
            .iter()
            .zip(negative.data())
            .all(|(actual, expected)| (*actual - *expected).norm() < 1.0e-10));
    }
}

#[test]
fn checked_generic_sqrt_rejects_shape_before_queries_and_preserves_source() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |_, _| 2.0).unwrap();
    let before = source.data().to_vec();
    reset_provider_queries(&provider);
    match source.sqrt() {
        Err(tenet::typed::Error::InvalidArgument(message)) => {
            assert!(message.contains("diagonal bond tensor"));
        }
        other => panic!("expected shape rejection, got {other:?}"),
    }
    assert_eq!(source.data(), before.as_slice());
    assert_no_provider_queries(&provider);

    let dense = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
        if indices[0] == indices[1] {
            4.0
        } else {
            0.0
        }
    })
    .unwrap();
    let before = dense.data().to_vec();
    reset_provider_queries(&provider);
    let root = dense.sqrt().unwrap();
    assert_no_provider_queries(&provider);
    assert!(root
        .compose(&root)
        .unwrap()
        .data()
        .iter()
        .zip(&before)
        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-12));
    assert_eq!(dense.data(), before.as_slice());

    macro_rules! assert_failure {
        ($fill:expr, $needle:literal) => {{
            let tensor = TensorMap::from_block_fn(&runtime, [&leg], [&leg], $fill).unwrap();
            let before = tensor.data().to_vec();
            reset_provider_queries(&provider);
            match tensor.sqrt() {
                Err(tenet::typed::Error::InvalidArgument(message)) => {
                    assert!(
                        message.contains($needle),
                        "unexpected sqrt error: {message}"
                    );
                }
                other => panic!("expected sqrt rejection, got {other:?}"),
            }
            assert_eq!(tensor.data(), before.as_slice());
            assert_no_provider_queries(&provider);
        }};
    }
    assert_failure!(
        |_, indices: &[usize]| {
            if indices[0] == indices[1] {
                -1.0
            } else {
                0.0
            }
        },
        "negative"
    );
    let bond_leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let tensor = TensorMap::from_block_fn(&runtime, [&bond_leg], [&bond_leg], |_, indices| {
        if indices[0] == indices[1] {
            4.0
        } else {
            1.0
        }
    })
    .unwrap();
    let before = tensor.data().to_vec();
    reset_provider_queries(&provider);
    match tensor.sqrt() {
        Err(tenet::typed::Error::InvalidArgument(message)) => {
            assert!(
                message.contains("off-diagonal"),
                "unexpected sqrt error: {message}"
            );
        }
        other => panic!("expected off-diagonal rejection, got {other:?}"),
    }
    assert_eq!(tensor.data(), before.as_slice());
    assert_no_provider_queries(&provider);
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_checked_generic_full_svd_preserves_provider_reconstructs_and_rejects_lazy() {
    use tenet::typed::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(vec![1, 1], 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |trees, _| {
            trees.coupled().iter().sum::<i64>() as f64 + 1.0
        })
        .unwrap();

    let (u, s, vh) = source.svd_full().unwrap();
    assert!(std::ptr::eq(u.provider(), provider.as_ref()));
    assert!(std::ptr::eq(s.provider(), provider.as_ref()));
    assert!(std::ptr::eq(vh.provider(), provider.as_ref()));
    let rebuilt = u.compose(&s).unwrap().compose(&vh).unwrap();
    assert!(rebuilt
        .data()
        .iter()
        .zip(source.data())
        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-10));

    let complex = source.to_c64();
    let (complex_u, complex_s, complex_vh) = complex.svd_full().unwrap();
    let complex_rebuilt = complex_u
        .compose(&complex_s)
        .unwrap()
        .compose(&complex_vh)
        .unwrap();
    assert!(complex_rebuilt
        .data()
        .iter()
        .zip(complex.data())
        .all(|(actual, expected)| (*actual - *expected).norm() < 1.0e-10));

    let lazy = source.adjoint().unwrap();
    assert!(matches!(
        lazy.svd_full(),
        Err(GenericTensorError::Facade(_))
    ));
}

#[cfg(feature = "racah-generated")]
fn assert_sun_checked_generic_inv<D>(n: usize, label: Vec<i64>)
where
    D: tenet::typed::TensorScalar + fmt::Debug + PartialEq,
{
    use tenet::typed::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(n).unwrap());
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(label, 2)]).unwrap();
    let source: TensorMap<_, D> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, indices| {
            let row = indices[0] + 2 * indices[1];
            let col = indices[2] + 2 * indices[3];
            D::from_real(
                if trees.codomain_vertices() == trees.domain_vertices() && row == col {
                    2.0
                } else {
                    0.0
                },
            )
        })
        .unwrap();
    assert!((0..source.block_count()).any(|index| {
        source
            .block_fusion_trees(index)
            .unwrap()
            .codomain_vertices()
            .iter()
            .chain(source.block_fusion_trees(index).unwrap().domain_vertices())
            .any(|vertex| vertex.get() > 1)
    }));
    let inverse = source.inv().unwrap();
    assert!(std::ptr::eq(inverse.provider(), provider.as_ref()));
    assert!(source.runtime().shares_state_with(inverse.runtime()));
    assert_eq!(inverse.codomain(), source.domain());
    assert_eq!(inverse.domain(), source.codomain());
    let expected = source.scale(D::from_real(0.5));
    for identity in [
        source.compose(&inverse).unwrap(),
        inverse.compose(&source).unwrap(),
    ] {
        assert_eq!(identity.data(), expected.data());
    }
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_checked_generic_inv_preserves_provider_outer_multiplicity_and_inverse_laws() {
    for (n, label) in [(3, vec![1, 1]), (4, vec![1, 0, 1])] {
        assert_sun_checked_generic_inv::<f64>(n, label.clone());
        assert_sun_checked_generic_inv::<Complex64>(n, label);
    }
}

#[cfg(feature = "racah-generated")]
fn assert_sun_checked_generic_left_solve(n: usize, label: Vec<i64>) {
    use tenet::typed::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(n).unwrap());
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(label, 2)]).unwrap();
    let divisor: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, indices| {
            let row = indices[0] + 2 * indices[1];
            let col = indices[2] + 2 * indices[3];
            if row == col {
                7.0 + trees.codomain_vertices()[0].get() as f64
                    + 0.25 * trees.domain_vertices()[0].get() as f64
            } else {
                0.05 * (1
                    + trees.codomain_vertices()[0].get()
                    + 2 * trees.domain_vertices()[0].get()) as f64
            }
        })
        .unwrap();
    assert!((0..divisor.block_count()).any(|index| {
        divisor
            .block_fusion_trees(index)
            .unwrap()
            .codomain_vertices()
            .iter()
            .chain(divisor.block_fusion_trees(index).unwrap().domain_vertices())
            .any(|vertex| vertex.get() > 1)
    }));
    let rhs: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, indices| {
            (indices.iter().sum::<usize>()
                + 1
                + 3 * trees.codomain_vertices()[0].get()
                + 5 * trees.domain_vertices()[0].get()) as f64
        })
        .unwrap();
    let route_swapped_rhs: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, indices| {
            (indices.iter().sum::<usize>()
                + 1
                + 3 * trees.domain_vertices()[0].get()
                + 5 * trees.codomain_vertices()[0].get()) as f64
        })
        .unwrap();

    let solution = divisor.solve(&rhs).unwrap();
    assert!(std::ptr::eq(solution.provider(), provider.as_ref()));
    let reconstructed = divisor.compose(&solution).unwrap();
    for index in 0..rhs.block_count() {
        assert_eq!(
            reconstructed.block_fusion_trees(index).unwrap(),
            rhs.block_fusion_trees(index).unwrap()
        );
        assert_eq!(
            reconstructed.block(index).unwrap().shape(),
            rhs.block(index).unwrap().shape()
        );
    }
    assert!(reconstructed
        .data()
        .iter()
        .zip(rhs.data())
        .all(|(actual, expected)| (*actual - *expected).abs() < 2e-10));
    assert!(reconstructed
        .data()
        .iter()
        .zip(route_swapped_rhs.data())
        .any(|(actual, swapped)| (*actual - *swapped).abs() > 1e-7));

    let complex_divisor = divisor.to_c64();
    let complex_rhs = rhs.to_c64().scale(Complex64::new(1.0, 0.25));
    let complex_solution = complex_divisor.solve(&complex_rhs).unwrap();
    let complex_reconstructed = complex_divisor.compose(&complex_solution).unwrap();
    for index in 0..complex_rhs.block_count() {
        assert_eq!(
            complex_reconstructed.block_fusion_trees(index).unwrap(),
            complex_rhs.block_fusion_trees(index).unwrap()
        );
        assert_eq!(
            complex_reconstructed.block(index).unwrap().shape(),
            complex_rhs.block(index).unwrap().shape()
        );
    }
    assert!(complex_reconstructed
        .data()
        .iter()
        .zip(complex_rhs.data())
        .all(|(actual, expected)| (*actual - *expected).norm() < 2e-10));
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_checked_generic_left_solve_preserves_outer_multiplicity() {
    for (n, label) in [(3, vec![1, 1]), (4, vec![1, 0, 1])] {
        assert_sun_checked_generic_left_solve(n, label);
    }
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_checked_generic_inv_preflight_counts_outer_multiplicity() {
    use tenet::typed::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let adjoint = vec![1, 1];
    let codomain_leg =
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(adjoint.clone(), 1)]).unwrap();
    let isomorphic_domain = GradedSpace::try_new_with_arc(
        Arc::clone(&provider),
        [
            (vec![0, 0], 1),
            (adjoint.clone(), 2),
            (vec![3, 0], 1),
            (vec![0, 3], 1),
            (vec![2, 2], 1),
        ],
    )
    .unwrap();
    let accepted: TensorMap<_, f64> = TensorMap::from_block_fn(
        &runtime,
        [&codomain_leg, &codomain_leg],
        [&isomorphic_domain],
        |trees, indices| {
            if trees.coupled() == &adjoint {
                let row = trees.codomain_vertices()[0].get() - 1;
                if row == indices[2] {
                    2.0
                } else {
                    0.0
                }
            } else {
                2.0
            }
        },
    )
    .unwrap();
    let inverse = accepted.inv().unwrap();
    assert_eq!(inverse.codomain(), accepted.domain());
    assert_eq!(inverse.domain(), accepted.codomain());

    let nonisomorphic_domain = GradedSpace::try_new_with_arc(
        Arc::clone(&provider),
        [
            (vec![0, 0], 1),
            (adjoint, 1),
            (vec![3, 0], 1),
            (vec![0, 3], 1),
            (vec![2, 2], 1),
        ],
    )
    .unwrap();
    let rejected: TensorMap<_, f64> = TensorMap::from_block_fn(
        &runtime,
        [&codomain_leg, &codomain_leg],
        [&nonisomorphic_domain],
        |_, _| 1.0,
    )
    .unwrap();
    let before = rejected.data().to_vec();
    assert!(matches!(
        rejected.inv(),
        Err(GenericTensorError::Facade(tenet::typed::Error::Operation(
            _
        )))
    ));
    assert_eq!(rejected.data(), before.as_slice());
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_checked_generic_compact_lq_preserves_provider_and_reconstructs() {
    use tenet::typed::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(vec![1, 1], 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |trees, _| {
            trees.coupled().iter().sum::<i64>() as f64 + 1.0
        })
        .unwrap();

    let (l, q) = source.lq_compact().unwrap();
    assert!(std::ptr::eq(l.provider(), provider.as_ref()));
    assert!(std::ptr::eq(q.provider(), provider.as_ref()));
    let rebuilt = l.compose(&q).unwrap();
    assert!(rebuilt
        .data()
        .iter()
        .zip(source.data())
        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-10));

    let complex = source.to_c64();
    let (complex_l, complex_q) = complex.lq_compact().unwrap();
    assert!(std::ptr::eq(complex_l.provider(), provider.as_ref()));
    assert!(std::ptr::eq(complex_q.provider(), provider.as_ref()));
    let complex_rebuilt = complex_l.compose(&complex_q).unwrap();
    assert!(complex_rebuilt
        .data()
        .iter()
        .zip(complex.data())
        .all(|(actual, expected)| (*actual - *expected).norm() < 1.0e-10));
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_checked_generic_orth_aliases_reconstruct_multiplicity_fixture() {
    use tenet::typed::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    macro_rules! assert_aliases {
        ($source:expr, $close:expr) => {{
            let source = $source;
            let close = $close;
            let (q, r) = source.qr_compact().unwrap();
            let (orth_q, orth_r) = source.left_orth().unwrap();
            for (actual, expected) in orth_q
                .compose(&orth_r)
                .unwrap()
                .data()
                .iter()
                .zip(source.data())
            {
                assert!(close(*actual, *expected) < 1.0e-10);
            }
            for (alias, lower) in [(&orth_q, &q), (&orth_r, &r)] {
                assert_eq!(alias.data(), lower.data());
                assert!(std::ptr::eq(alias.provider(), lower.provider()));
                assert_eq!(alias.codomain(), lower.codomain());
                assert_eq!(alias.domain(), lower.domain());
                assert!(alias.runtime().shares_state_with(lower.runtime()));
            }

            let (l, q) = source.lq_compact().unwrap();
            let (orth_l, orth_q) = source.right_orth().unwrap();
            for (actual, expected) in orth_l
                .compose(&orth_q)
                .unwrap()
                .data()
                .iter()
                .zip(source.data())
            {
                assert!(close(*actual, *expected) < 1.0e-10);
            }
            for (alias, lower) in [(&orth_l, &l), (&orth_q, &q)] {
                assert_eq!(alias.data(), lower.data());
                assert!(std::ptr::eq(alias.provider(), lower.provider()));
                assert_eq!(alias.codomain(), lower.codomain());
                assert_eq!(alias.domain(), lower.domain());
                assert!(alias.runtime().shares_state_with(lower.runtime()));
            }
        }};
    }

    for n in [3, 4] {
        let provider = Arc::new(SUNFusionRule::new(n).unwrap());
        let label = if n == 3 { vec![1, 1] } else { vec![1, 0, 1] };
        let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(label, 1)]).unwrap();
        let source: TensorMap<_, f64> =
            TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |trees, _| {
                trees.codomain_vertices()[0].get() as f64
            })
            .unwrap();
        assert_eq!(source.block_count(), 2);
        assert_aliases!(&source, |actual: f64, expected: f64| {
            (actual - expected).abs()
        });
        assert_aliases!(source.to_c64(), |actual: Complex64, expected: Complex64| {
            (actual - expected).norm()
        });
    }
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_checked_generic_full_qr_preserves_provider_and_reconstructs() {
    use tenet::typed::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(vec![1, 1], 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |trees, _| {
            trees.coupled().iter().sum::<i64>() as f64 + 1.0
        })
        .unwrap();

    let (q, r) = source.qr_full().unwrap();
    assert!(std::ptr::eq(q.provider(), provider.as_ref()));
    assert!(std::ptr::eq(r.provider(), provider.as_ref()));
    let rebuilt = q.compose(&r).unwrap();
    assert!(rebuilt
        .data()
        .iter()
        .zip(source.data())
        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-10));

    let complex = source.to_c64();
    let (complex_q, complex_r) = complex.qr_full().unwrap();
    assert!(std::ptr::eq(complex_q.provider(), provider.as_ref()));
    assert!(std::ptr::eq(complex_r.provider(), provider.as_ref()));
    let complex_rebuilt = complex_q.compose(&complex_r).unwrap();
    assert!(complex_rebuilt
        .data()
        .iter()
        .zip(complex.data())
        .all(|(actual, expected)| (*actual - *expected).norm() < 1.0e-10));
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_checked_generic_svd_vals_matches_compact_spectrum() {
    use tenet::typed::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(vec![1, 1], 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |trees, _| {
            trees.coupled().iter().sum::<i64>() as f64 + 1.0
        })
        .unwrap();
    let spectra = source.svd_vals().unwrap();
    assert!(!spectra.is_empty());
    assert!(spectra.iter().all(|spectrum| spectrum
        .values
        .iter()
        .all(|value| value.is_finite() && *value >= 0.0)));

    let complex = source.to_c64();
    let complex_spectra = complex.svd_vals().unwrap();
    assert_eq!(complex_spectra, spectra);
}

#[test]
fn checked_generic_eigh_vals_preserves_spectrum_and_dtype() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, _| 2.5).unwrap();

    let spectra = source.eigh_vals().unwrap();
    assert_eq!(spectra.len(), 1);
    assert_eq!(spectra[0].sector, Label::X);
    assert_eq!(spectra[0].values, vec![2.5]);

    let complex = source.to_c64();
    assert_eq!(complex.eigh_vals().unwrap(), spectra);
}

fn assert_checked_generic_eigh_factors<D>(
    source: &TensorMap<CheckedOnlyToy, D>,
    close: impl Fn(D, D) -> f64 + Copy,
    adjoint: impl Fn(D) -> D + Copy,
) where
    D: tenet::typed::TensorScalar + fmt::Debug,
{
    let (d, v) = source.eigh_full().unwrap();
    assert!(std::ptr::eq(d.provider(), source.provider()));
    assert!(std::ptr::eq(v.provider(), source.provider()));
    assert!(d.runtime().shares_state_with(source.runtime()));
    assert!(v.runtime().shares_state_with(source.runtime()));
    assert_eq!(v.codomain(), source.codomain());
    assert_eq!(d.codomain(), d.domain());
    assert_eq!(v.domain(), d.codomain());
    assert!(format!("{d:?}").contains("elements: 3"));
    assert_eq!(d.data().len(), 9);

    for (actual, expected) in source
        .compose(&v)
        .unwrap()
        .data()
        .iter()
        .zip(v.compose(&d).unwrap().data())
    {
        assert!(close(*actual, *expected) < 1e-10);
    }
    let vectors = v.data();
    let diagonal = d.data();
    for column in 0..3 {
        for row in 0..3 {
            let gram = (0..3).fold(D::from_real(0.0), |sum, inner| {
                sum + adjoint(vectors[inner + row * 3]) * vectors[inner + column * 3]
            });
            assert!(close(gram, D::from_real(f64::from(row == column))) < 1e-10);
            let rebuilt = (0..3).fold(D::from_real(0.0), |sum, inner| {
                sum + vectors[row + inner * 3]
                    * diagonal[inner + inner * 3]
                    * adjoint(vectors[column + inner * 3])
            });
            assert!(close(rebuilt, source.data()[row + column * 3]) < 1e-10);
        }
    }

    let truncated = source.eigh_trunc(&Truncation::rank(5)).unwrap();
    assert!(std::ptr::eq(truncated.d.provider(), source.provider()));
    assert!(std::ptr::eq(truncated.v.provider(), source.provider()));
    assert_eq!(truncated.eigenvalues.len(), 1);
    assert_eq!(truncated.eigenvalues[0].sector, Label::X);
    assert_eq!(truncated.eigenvalues[0].values, vec![-3.0, 2.0]);
    let expected_error = (1.0 + 2.0_f64.sqrt()).sqrt();
    assert!((truncated.error - expected_error).abs() < 1e-12);
}

#[test]
fn checked_generic_eigh_full_and_trunc_preserve_contract_for_both_dtypes() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 3)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, index| {
            [-3.0, 2.0, 1.0][index[0]] * f64::from(index[0] == index[1])
        })
        .unwrap();
    assert_checked_generic_eigh_factors(
        &source,
        |actual, expected| (actual - expected).abs(),
        |value| value,
    );
    let complex = source.to_c64();
    assert_checked_generic_eigh_factors(
        &complex,
        |actual, expected| (actual - expected).norm(),
        |value| value.conj(),
    );
    let (complex_d, _) = complex.eigh_full().unwrap();
    assert!(complex_d.data().iter().all(|value| value.im == 0.0));
}

#[test]
fn checked_generic_eigh_lazy_success_and_failure_leave_the_view_lazy() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let hermitian: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, index| {
            [[2.0, 1.0], [1.0, -1.0]][index[0]][index[1]]
        })
        .unwrap();
    let lazy = hermitian.adjoint().unwrap();
    assert!(lazy.network_reuse_class(false) == tenet::typed::NetworkReuseClass::LazyAdjoint);
    assert!(lazy.eigh_full().is_ok());
    assert!(lazy.eigh_trunc(&Truncation::rank(1)).is_ok());
    assert!(lazy.network_reuse_class(false) == tenet::typed::NetworkReuseClass::LazyAdjoint);

    let nonhermitian: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, index| {
            [[0.0, 1.0], [0.0, 0.0]][index[0]][index[1]]
        })
        .unwrap();
    let lazy = nonhermitian.adjoint().unwrap();
    assert!(lazy.eigh_full().is_err());
    assert!(lazy.eigh_trunc(&Truncation::Full).is_err());
    assert!(lazy.network_reuse_class(false) == tenet::typed::NetworkReuseClass::LazyAdjoint);
}

#[test]
fn checked_generic_eigh_rejects_invalid_inputs_before_publication() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let wide = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let narrow = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let nonendomorphism: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&wide], [&narrow], |_, _| 1.0).unwrap();
    assert!(nonendomorphism.eigh_full().is_err());

    let nonhermitian: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&wide], [&wide], |_, index| {
            [[0.0, 1.0], [0.0, 0.0]][index[0]][index[1]]
        })
        .unwrap();
    assert!(nonhermitian.eigh_full().is_err());
    let nonfinite: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&wide], [&wide], |_, index| {
            if index[0] == index[1] {
                f64::NAN
            } else {
                0.0
            }
        })
        .unwrap();
    assert!(nonfinite.eigh_full().is_err());
}

struct EighFaultExecutor {
    inner: DefaultDenseExecutor,
    calls: Arc<AtomicUsize>,
    fail_at: Option<usize>,
}

#[derive(Clone, Copy)]
enum EigFault {
    Eig,
    Svd,
}

struct EigFaultExecutor {
    inner: DefaultDenseExecutor,
    fault: EigFault,
}

impl DenseExecutor for EigFaultExecutor {
    fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.svd(input)
    }

    fn svd_vals(&mut self, input: DenseRead<'_>) -> Result<DenseTensor, DenseError> {
        if matches!(self.fault, EigFault::Svd) {
            return Err(DenseError::Backend {
                backend: DenseBackend::Tenferro,
                op: "svd_vals",
                message: "injected checked Generic EIG rank-check failure".to_string(),
            });
        }
        self.inner.svd_vals(input)
    }

    fn qr(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.qr(input)
    }

    fn eigh(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.eigh(input)
    }

    fn eig(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        if matches!(self.fault, EigFault::Eig) {
            return Err(DenseError::Backend {
                backend: DenseBackend::Tenferro,
                op: "eig",
                message: "injected checked Generic EIG failure".to_string(),
            });
        }
        self.inner.eig(input)
    }

    fn dot_general_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        config: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        self.inner.dot_general_into(output, lhs, rhs, config)
    }
}

impl DenseExecutor for EighFaultExecutor {
    fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.svd(input)
    }

    fn qr(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.qr(input)
    }

    fn eigh(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.eigh(input)
    }

    fn eigh_into(
        &mut self,
        input: DenseRead<'_>,
        values: DenseWrite<'_>,
        vectors: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        let call = self.calls.fetch_add(1, Ordering::Relaxed) + 1;
        if self.fail_at == Some(call) {
            return Err(DenseError::Backend {
                backend: DenseBackend::Tenferro,
                op: "eigh_into",
                message: "injected checked Generic EIGH failure".to_string(),
            });
        }
        self.inner.eigh_into(input, values, vectors)
    }

    fn dot_general_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        config: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        self.inner.dot_general_into(output, lhs, rhs, config)
    }
}

#[test]
fn checked_generic_eigh_preflights_all_sectors_and_runs_once_per_sector() {
    let calls = Arc::new(AtomicUsize::new(0));
    let runtime = Runtime::builder()
        .dense_threads(1)
        .with_dense_executor(Box::new(EighFaultExecutor {
            inner: DefaultDenseExecutor::new(),
            calls: Arc::clone(&calls),
            fail_at: None,
        }))
        .build()
        .unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg =
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::Vacuum, 1), (Label::X, 2)])
            .unwrap();
    let hermitian: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |trees, index| {
            if index[0] == index[1] {
                if trees.coupled() == &Label::Vacuum {
                    1.0
                } else {
                    2.0
                }
            } else {
                0.0
            }
        })
        .unwrap();
    hermitian.eigh_full().unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 2);

    calls.store(0, Ordering::Relaxed);
    let nonhermitian: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |trees, index| {
            if trees.coupled() == &Label::X && index == [0, 1] {
                1.0
            } else {
                0.0
            }
        })
        .unwrap();
    assert!(nonhermitian.eigh_full().is_err());
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[test]
fn checked_generic_eigh_dense_failure_preserves_the_source() {
    let calls = Arc::new(AtomicUsize::new(0));
    let runtime = Runtime::builder()
        .dense_threads(1)
        .with_dense_executor(Box::new(EighFaultExecutor {
            inner: DefaultDenseExecutor::new(),
            calls: Arc::clone(&calls),
            fail_at: Some(1),
        }))
        .build()
        .unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, index| {
            [[2.0, 1.0], [1.0, -1.0]][index[0]][index[1]]
        })
        .unwrap();
    let before = source.data().to_vec();
    assert!(source.eigh_full().is_err());
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(source.data(), before);
    assert!(std::ptr::eq(source.provider(), provider.as_ref()));
}

#[test]
fn checked_generic_eig_dense_and_rank_check_failures_preserve_the_source() {
    for fault in [EigFault::Eig, EigFault::Svd] {
        let runtime = Runtime::builder()
            .dense_threads(1)
            .with_dense_executor(Box::new(EigFaultExecutor {
                inner: DefaultDenseExecutor::new(),
                fault,
            }))
            .build()
            .unwrap();
        let provider = Arc::new(CheckedOnlyToy::new(0));
        let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
        let source: TensorMap<_, f64> =
            TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, index| {
                [[1.0, -3.0], [1.0, 1.0]][index[0]][index[1]]
            })
            .unwrap();
        let before = source.data().to_vec();
        assert!(source.eig_full().is_err());
        assert_eq!(source.data(), before);
        assert!(std::ptr::eq(source.provider(), provider.as_ref()));
    }
}

#[test]
fn checked_generic_eigh_qdim_and_decode_failures_publish_no_pair() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, index| {
            [2.0, -1.0][index[0]] * f64::from(index[0] == index[1])
        })
        .unwrap();
    let before = source.data().to_vec();

    provider.fail_decode.store(true, Ordering::Relaxed);
    assert!(source.eigh_full().is_ok());
    assert!(matches!(
        source.eigh_trunc(&Truncation::Full),
        Err(GenericTensorError::Plan(
            tenet::typed::CheckedGenericPlanError::Provider(ToyError::Decode)
        ))
    ));
    provider.fail_decode.store(false, Ordering::Relaxed);

    provider.fail_dim.store(true, Ordering::Relaxed);
    assert!(matches!(
        source.eigh_trunc(&Truncation::rank(1)),
        Err(GenericTensorError::Plan(
            tenet::typed::CheckedGenericPlanError::Provider(ToyError::Algebra)
        ))
    ));
    provider.fail_dim.store(false, Ordering::Relaxed);

    assert_eq!(source.data(), before);
    assert!(std::ptr::eq(source.provider(), provider.as_ref()));
}

#[test]
fn checked_generic_eig_qdim_and_decode_failures_publish_no_pair() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, index| {
            [[1.0, -3.0], [1.0, 1.0]][index[0]][index[1]]
        })
        .unwrap();
    let before = source.data().to_vec();

    provider.fail_decode.store(true, Ordering::Relaxed);
    assert!(source.eig_full().is_ok());
    assert!(matches!(
        source.eig_trunc(&Truncation::Full),
        Err(GenericTensorError::Plan(
            tenet::typed::CheckedGenericPlanError::Provider(ToyError::Decode)
        ))
    ));
    provider.fail_decode.store(false, Ordering::Relaxed);

    provider.fail_dim.store(true, Ordering::Relaxed);
    assert!(matches!(
        source.eig_trunc(&Truncation::rank(1)),
        Err(GenericTensorError::Plan(
            tenet::typed::CheckedGenericPlanError::Provider(ToyError::Algebra)
        ))
    ));
    provider.fail_dim.store(false, Ordering::Relaxed);

    assert_eq!(source.data(), before);
    assert!(std::ptr::eq(source.provider(), provider.as_ref()));
}

#[test]
fn checked_generic_eigh_signed_ties_are_stable_and_degenerate_projectors_are_invariant() {
    use std::cell::Cell;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 3)]).unwrap();
    let tied: TensorMap<_, f64> = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, index| {
        [-2.0, 2.0, 1.0][index[0]] * f64::from(index[0] == index[1])
    })
    .unwrap();
    let (d, _) = tied.eigh_full().unwrap();
    assert_eq!([d.data()[0], d.data()[4], d.data()[8]], [-2.0, 2.0, 1.0]);

    let degenerate: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, index| {
            [2.0, 2.0, -1.0][index[0]] * f64::from(index[0] == index[1])
        })
        .unwrap();
    let (d, v) = degenerate.eigh_full().unwrap();
    let selector_codomain = d.codomain();
    let selector_domain = d.domain();
    let selector: TensorMap<_, f64> = TensorMap::from_block_fn(
        &runtime,
        selector_codomain.iter(),
        selector_domain.iter(),
        |_, index| f64::from(index[0] == index[1] && d.data()[index[0] * 4] == 2.0),
    )
    .unwrap();
    let lazy_vh = v.adjoint().unwrap();
    let data = lazy_vh.data().to_vec();
    let position = Cell::new(0usize);
    let codomain = v.domain();
    let domain = v.codomain();
    let vh: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, codomain.iter(), domain.iter(), |_, _| {
            let index = position.get();
            position.set(index + 1);
            data[index]
        })
        .unwrap();
    let projector = v.compose(&selector).unwrap().compose(&vh).unwrap();
    assert_eq!(
        projector.data(),
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    );
}

#[test]
fn checked_generic_eig_vals_preserves_spectrum_and_dtype() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            if indices[0] == indices[1] {
                (indices[0] + 2) as f64
            } else {
                0.0
            }
        })
        .unwrap();

    let spectra = source.eig_vals().unwrap();
    assert_eq!(spectra.len(), 1);
    assert_eq!(spectra[0].sector, Label::X);
    assert_eq!(
        spectra[0].values,
        vec![Complex64::new(3.0, 0.0), Complex64::new(2.0, 0.0)]
    );

    let complex = source.to_c64();
    assert_eq!(complex.eig_vals().unwrap(), spectra);
}

fn assert_checked_generic_eig_reconstruction(
    source: &TensorMap<CheckedOnlyToy, Complex64>,
    d: &TensorMap<CheckedOnlyToy, Complex64>,
    v: &TensorMap<CheckedOnlyToy, Complex64>,
) {
    assert!(std::ptr::eq(d.provider(), source.provider()));
    assert!(std::ptr::eq(v.provider(), source.provider()));
    assert!(d.runtime().shares_state_with(source.runtime()));
    assert!(v.runtime().shares_state_with(source.runtime()));
    assert_eq!(v.codomain(), source.codomain());
    assert_eq!(v.domain(), d.codomain());
    let av = source.compose(v).unwrap();
    let vd = v.compose(d).unwrap();
    let scale = source
        .data()
        .iter()
        .map(|value| value.norm())
        .fold(1.0_f64, f64::max);
    assert!(av
        .data()
        .iter()
        .zip(vd.data())
        .all(|(actual, expected)| (*actual - *expected).norm() <= 1.0e-11 * scale));
    let rebuilt = v.compose(d).unwrap().compose(&v.inv().unwrap()).unwrap();
    assert_same_checked_generic_layout_and_close(&rebuilt, source, |actual, expected| {
        (actual - expected).norm()
    });
}

#[test]
fn checked_generic_eig_full_is_complex_and_reconstructs_nonnormal_inputs() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let real: TensorMap<_, f64> = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, index| {
        [[1.0, -3.0], [1.0, 1.0]][index[0]][index[1]]
    })
    .unwrap();
    let (real_d, real_v) = real.eig_full().unwrap();
    assert!(real_d.data().iter().any(|value| value.im.abs() > 1.0));
    for column in 0..2 {
        let pivot = (0..2)
            .max_by(|&left, &right| {
                real_v.data()[left + column * 2]
                    .norm()
                    .total_cmp(&real_v.data()[right + column * 2].norm())
            })
            .unwrap();
        let pivot = real_v.data()[pivot + column * 2];
        assert!(pivot.im.abs() < 1.0e-12);
        assert!(pivot.re >= 0.0);
    }
    assert_checked_generic_eig_reconstruction(&real.to_c64(), &real_d, &real_v);

    let complex = real.to_c64().scale(Complex64::new(1.0, 0.25));
    let (complex_d, complex_v) = complex.eig_full().unwrap();
    assert!(complex_d.data().iter().any(|value| value.im.abs() > 1.0));
    assert_checked_generic_eig_reconstruction(&complex, &complex_d, &complex_v);
}

#[test]
fn checked_generic_eig_ties_are_stable_and_degenerate_projectors_are_invariant() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 3)]).unwrap();
    let tied: TensorMap<_, f64> = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, index| {
        [-2.0, 2.0, 1.0][index[0]] * f64::from(index[0] == index[1])
    })
    .unwrap();
    let (d, _) = tied.eig_full().unwrap();
    assert_eq!(
        [d.data()[0], d.data()[4], d.data()[8]],
        [
            Complex64::new(-2.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(1.0, 0.0)
        ]
    );

    let degenerate: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, index| {
            [2.0, 2.0, -1.0][index[0]] * f64::from(index[0] == index[1])
        })
        .unwrap();
    let (d, v) = degenerate.eig_full().unwrap();
    let selector: TensorMap<_, Complex64> = TensorMap::from_block_fn(
        &runtime,
        d.codomain().iter(),
        d.domain().iter(),
        |_, index| {
            Complex64::new(
                f64::from(
                    index[0] == index[1]
                        && (d.data()[index[0] * 4] - Complex64::new(2.0, 0.0)).norm() < 1.0e-12,
                ),
                0.0,
            )
        },
    )
    .unwrap();
    let projector = v
        .compose(&selector)
        .unwrap()
        .compose(&v.inv().unwrap())
        .unwrap();
    assert_eq!(
        projector.data(),
        &[
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]
    );
}

#[test]
fn checked_generic_eig_trunc_reports_discarded_spectrum_norm_only() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 3)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, index| {
            [-3.0, 2.0, 1.0][index[0]] * f64::from(index[0] == index[1])
        })
        .unwrap();
    let truncated = source.eig_trunc(&Truncation::rank(5)).unwrap();
    assert!(std::ptr::eq(truncated.d.provider(), provider.as_ref()));
    assert!(std::ptr::eq(truncated.v.provider(), provider.as_ref()));
    assert_eq!(truncated.eigenvalues[0].sector, Label::X);
    assert_eq!(
        truncated.eigenvalues[0].values,
        vec![Complex64::new(-3.0, 0.0), Complex64::new(2.0, 0.0)]
    );
    assert!((truncated.error - (1.0 + 2.0_f64.sqrt()).sqrt()).abs() < 1.0e-12);
}

#[test]
fn checked_generic_eig_rejects_jordan_and_nonfinite_inputs() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let jordan: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, index| {
            [[1.0, 1.0], [0.0, 1.0]][index[0]][index[1]]
        })
        .unwrap();
    let error = jordan.eig_full().unwrap_err();
    assert!(format!("{error:?}")
        .contains("eig requires a numerically diagonalizable coupled-sector matrix"));

    let nonfinite: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, index| {
            if index == [0, 0] {
                f64::NAN
            } else {
                0.0
            }
        })
        .unwrap();
    let error = nonfinite.eig_full().unwrap_err();
    assert!(format!("{error:?}").contains("eig input components must be finite"));
}

#[test]
fn checked_generic_eig_lazy_calls_leave_the_source_view_lazy() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, index| {
            [[1.0, -3.0], [1.0, 1.0]][index[0]][index[1]]
        })
        .unwrap();
    let lazy = source.adjoint().unwrap();
    assert!(lazy.network_reuse_class(false) == tenet::typed::NetworkReuseClass::LazyAdjoint);
    assert!(lazy.eig_full().is_ok());
    assert!(lazy.eig_trunc(&Truncation::rank(1)).is_ok());
    assert!(lazy.network_reuse_class(false) == tenet::typed::NetworkReuseClass::LazyAdjoint);
}

#[test]
fn checked_generic_svd_trunc_reconstructs_and_preserves_provider() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            if indices[0] == indices[1] {
                (indices[0] + 2) as f64
            } else {
                0.0
            }
        })
        .unwrap();
    let result = source.svd_trunc(&Truncation::rank(1)).unwrap();
    assert!(std::ptr::eq(result.u.provider(), provider.as_ref()));
    assert!(std::ptr::eq(result.s.provider(), provider.as_ref()));
    assert!(std::ptr::eq(result.vh.provider(), provider.as_ref()));
    let rebuilt = result
        .u
        .compose(&result.s)
        .unwrap()
        .compose(&result.vh)
        .unwrap();
    assert!(rebuilt.data().iter().all(|value| value.is_finite()));
    assert!(result.singular_values.iter().all(|spectrum| {
        spectrum.values.len() <= 2 && spectrum.values.iter().all(|value| value.is_finite())
    }));
}

#[test]
fn checked_generic_lazy_adjoint_preserves_provider_and_reductions() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            (indices.iter().sum::<usize>() + 1) as f64
        })
        .unwrap();

    let adjoint = source.adjoint().unwrap();
    assert!(std::ptr::eq(adjoint.provider(), provider.as_ref()));
    assert!((adjoint.norm().unwrap() - source.norm().unwrap()).abs() < 1.0e-12);
    assert!((adjoint.tr().unwrap() - source.tr().unwrap()).abs() < 1.0e-12);

    let complex = source.to_c64();
    let complex_adjoint = complex.adjoint().unwrap();
    assert!(std::ptr::eq(complex_adjoint.provider(), provider.as_ref()));
    assert!((complex_adjoint.tr().unwrap() - complex.tr().unwrap().conj()).norm() < 1.0e-12);
}

#[test]
fn checked_generic_exp_uses_general_pade_for_nonhermitian_dense_blocks() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, ij| f64::from(ij == [0, 1]))
            .unwrap();
    let expected = [1.0, 0.0, 1.0, 1.0];
    reset_provider_queries(&provider);
    let direct = source.exp().unwrap();
    assert_no_provider_queries(&provider);
    assert!(std::ptr::eq(direct.provider(), provider.as_ref()));
    assert!(direct.runtime().shares_state_with(source.runtime()));
    assert_eq!(direct.codomain(), source.codomain());
    assert_eq!(direct.domain(), source.domain());
    assert_eq!(direct.block_count(), source.block_count());
    assert!(direct
        .data()
        .iter()
        .zip(expected)
        .all(|(a, b)| (*a - b).abs() < 1e-12));
    let lazy = source.adjoint().unwrap();
    let lazy_exp = lazy.exp().unwrap();
    assert!(std::ptr::eq(lazy_exp.provider(), provider.as_ref()));
    assert!(lazy_exp.runtime().shares_state_with(source.runtime()));
    assert_eq!(lazy_exp.codomain(), lazy.codomain());
    assert_eq!(lazy_exp.domain(), lazy.domain());
    assert_eq!(lazy_exp.block_count(), lazy.block_count());
    assert_eq!(lazy_exp.data(), direct.adjoint().unwrap().data());
    let complex = source.to_c64().scale(Complex64::new(1.0, 0.25));
    reset_provider_queries(&provider);
    let complex_exp = complex.exp().unwrap();
    assert_no_provider_queries(&provider);
    assert!(std::ptr::eq(complex_exp.provider(), provider.as_ref()));
    assert!(complex_exp.runtime().shares_state_with(source.runtime()));
    assert_eq!(complex_exp.codomain(), complex.codomain());
    assert_eq!(complex_exp.domain(), complex.domain());
    assert!(complex_exp
        .data()
        .iter()
        .zip([
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.25),
            Complex64::new(1.0, 0.0)
        ])
        .all(|(a, b)| (*a - b).norm() < 1e-12));
}

#[test]
fn checked_generic_exp_rejects_nonendomorphism_before_provider_work() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let wide = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let narrow = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&wide], [&narrow], |_, _| 1.0).unwrap();
    let before = source.data().to_vec();
    reset_provider_queries(&provider);
    assert!(matches!(
        source.exp(),
        Err(GenericTensorError::Facade(tenet::typed::Error::Operation(
            _
        )))
    ));
    assert_no_provider_queries(&provider);
    assert_eq!(source.data(), before.as_slice());
}

#[test]
fn checked_generic_exp_rejects_early_and_late_nonfinite_sectors_without_publication() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg =
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::Vacuum, 1), (Label::X, 1)])
            .unwrap();
    for target in [Label::Vacuum, Label::X] {
        let source: TensorMap<_, f64> =
            TensorMap::from_block_fn(&runtime, [&leg], [&leg], |trees, _| {
                if trees.coupled() == &target {
                    f64::NAN
                } else {
                    0.0
                }
            })
            .unwrap();
        let before = source.data().to_vec();
        assert!(matches!(
            source.exp(),
            Err(GenericTensorError::Facade(tenet::typed::Error::Operation(
                _
            )))
        ));
        assert_eq!(source.data().len(), before.len());
        assert!(source
            .data()
            .iter()
            .zip(&before)
            .all(|(a, b)| a.to_bits() == b.to_bits()));
    }
}

#[cfg(feature = "racah-generated")]
fn assert_sun_checked_generic_exp_outer_multiplicity(n: usize, adjoint: Vec<i64>) {
    use tenet::typed::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(n).unwrap());
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(adjoint.clone(), 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, ij| {
            let left = trees.codomain_vertices()[0].get();
            let right = trees.domain_vertices()[0].get();
            if ij.iter().all(|&index| index == 0) {
                if left == right {
                    left as f64 / 5.0
                } else if left == 1 && right == 2 {
                    0.3
                } else {
                    0.0
                }
            } else {
                0.0
            }
        })
        .unwrap();
    assert!(
        (0..source.block_count()).any(|index| source
            .block_fusion_trees(index)
            .unwrap()
            .codomain_vertices()[0]
            .get()
            > 1),
        "[adj, adj] -> adj must retain its outer-multiplicity key"
    );
    let real_output = source.exp().unwrap();
    assert!(std::ptr::eq(real_output.provider(), provider.as_ref()));
    assert!(real_output.runtime().shares_state_with(source.runtime()));
    assert_eq!(real_output.codomain(), source.codomain());
    assert_eq!(real_output.domain(), source.domain());
    for index in 0..source.block_count() {
        assert_eq!(
            real_output.block_fusion_trees(index).unwrap(),
            source.block_fusion_trees(index).unwrap()
        );
        assert_eq!(
            real_output.block(index).unwrap(),
            source.block(index).unwrap()
        );
    }
    let real_inverse = source.scale(-1.0).exp().unwrap();
    let real_identity: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, _| {
            f64::from(trees.codomain_vertices() == trees.domain_vertices())
        })
        .unwrap();
    for product in [
        real_output.compose(&real_inverse).unwrap(),
        real_inverse.compose(&real_output).unwrap(),
    ] {
        assert!(product
            .data()
            .iter()
            .zip(real_identity.data())
            .all(|(a, b)| (*a - *b).abs() < 2e-10));
    }
    let input = source.to_c64().scale(Complex64::new(1.0, 0.2));
    let output = input.exp().unwrap();
    assert!(std::ptr::eq(output.provider(), provider.as_ref()));
    assert!(output.runtime().shares_state_with(source.runtime()));
    assert_eq!(output.codomain(), input.codomain());
    assert_eq!(output.domain(), input.domain());
    assert_eq!(output.block_count(), input.block_count());
    for index in 0..input.block_count() {
        assert_eq!(
            output.block_fusion_trees(index).unwrap(),
            input.block_fusion_trees(index).unwrap()
        );
        assert_eq!(output.block(index).unwrap(), input.block(index).unwrap());
    }
    let inverse = input.scale(Complex64::new(-1.0, 0.0)).exp().unwrap();
    let identity: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, _| {
            Complex64::new(
                f64::from(trees.codomain_vertices() == trees.domain_vertices()),
                0.0,
            )
        })
        .unwrap();
    for product in [
        output.compose(&inverse).unwrap(),
        inverse.compose(&output).unwrap(),
    ] {
        assert!(product
            .data()
            .iter()
            .zip(identity.data())
            .all(|(a, b)| (*a - *b).norm() < 2e-10));
    }
}

#[cfg(feature = "racah-generated")]
#[test]
fn checked_generic_exp_sun_outer_multiplicity_preserves_layout() {
    assert_sun_checked_generic_exp_outer_multiplicity(3, vec![1, 1]);
    assert_sun_checked_generic_exp_outer_multiplicity(4, vec![1, 0, 1]);
}

#[test]
fn checked_generic_reduction_dimension_failure_is_typed_and_nonpublishing() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, _| 2.0).unwrap();
    let before = source.data().to_vec();
    provider.fail_algebra.store(true, Ordering::Relaxed);
    let error = source.norm().unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Structure(CheckedGenericStructureError::Provider(ToyError::Algebra))
    ));
    assert_eq!(source.data(), before.as_slice());
}

#[test]
fn checked_generic_inv_isomorphism_preflight_failure_is_typed_and_nonpublishing() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, _| 2.0).unwrap();
    let before = source.data().to_vec();
    provider.fail_algebra.store(true, Ordering::Relaxed);
    assert!(matches!(
        source.inv(),
        Err(GenericTensorError::Plan(
            tenet::typed::CheckedGenericPlanError::Provider(ToyError::Algebra)
        ))
    ));
    assert_eq!(source.data(), before.as_slice());
}

#[test]
fn checked_generic_inv_destination_admission_failure_is_typed_and_nonpublishing() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, _| 2.0).unwrap();
    let before = source.data().to_vec();
    provider.invalid_style.store(true, Ordering::Relaxed);
    assert!(matches!(
        source.inv(),
        Err(GenericTensorError::Structure(_))
    ));
    assert_eq!(source.data(), before.as_slice());
}

#[test]
fn checked_generic_inv_accepts_unequal_isomorphic_spaces_and_rejects_nonisomorphic() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let x = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let unit = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::Vacuum, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&x], [&x, &unit], |_, indices| {
            if indices[0] == indices[1] {
                2.0
            } else {
                0.0
            }
        })
        .unwrap();
    let inverse = source.inv().unwrap();
    assert_eq!((inverse.codomain_rank(), inverse.domain_rank()), (2, 1));
    assert_eq!(inverse.codomain(), source.domain());
    assert_eq!(inverse.domain(), source.codomain());
    assert!(std::ptr::eq(inverse.provider(), provider.as_ref()));
    assert!(source.runtime().shares_state_with(inverse.runtime()));
    assert_eq!(
        source.compose(&inverse).unwrap().data(),
        source.scale(0.5).data()
    );
    assert_eq!(
        inverse.compose(&source).unwrap().data(),
        source.scale(0.5).data()
    );

    let narrow = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let nonisomorphic: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&narrow], [&x], |_, _| 1.0).unwrap();
    let before = nonisomorphic.data().to_vec();
    assert!(matches!(
        nonisomorphic.inv(),
        Err(GenericTensorError::Facade(tenet::typed::Error::Operation(
            _
        )))
    ));
    assert_eq!(nonisomorphic.data(), before.as_slice());
}

#[test]
fn checked_generic_pinv_rectangular_moore_penrose_and_validation_precedence() {
    macro_rules! assert_moore_penrose {
        ($input:expr, $pseudo:expr, $distance:expr) => {{
            let input = $input;
            let pseudo = $pseudo;
            let distance = $distance;
            let aa_plus = input.compose(pseudo).unwrap();
            let a_plus_a = pseudo.compose(input).unwrap();
            for (actual, expected) in aa_plus
                .compose(input)
                .unwrap()
                .data()
                .iter()
                .zip(input.data())
            {
                assert!(distance(*actual, *expected) < 1e-10);
            }
            for (actual, expected) in a_plus_a
                .compose(pseudo)
                .unwrap()
                .data()
                .iter()
                .zip(pseudo.data())
            {
                assert!(distance(*actual, *expected) < 1e-10);
            }
            for (actual, expected) in aa_plus.adjoint().unwrap().data().iter().zip(aa_plus.data()) {
                assert!(distance(*actual, *expected) < 1e-10);
            }
            for (actual, expected) in a_plus_a
                .adjoint()
                .unwrap()
                .data()
                .iter()
                .zip(a_plus_a.data())
            {
                assert!(distance(*actual, *expected) < 1e-10);
            }
        }};
    }

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let codomain = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let domain = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 3)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&codomain], [&domain], |_, index| {
            [[1.0, 0.0, 1.0], [0.0, 2.0, 1.0]][index[0]][index[1]]
        })
        .unwrap();
    let before = source.data().to_vec();
    reset_provider_queries(&provider);
    for rcond in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            source.pinv(rcond),
            Err(GenericTensorError::Facade(
                tenet::typed::Error::InvalidArgument(_)
            ))
        ));
        assert_no_provider_queries(&provider);
    }
    assert_eq!(source.data(), before.as_slice());

    let pseudo = source.pinv(1e-12).unwrap();
    assert_eq!(pseudo.codomain(), source.domain());
    assert_eq!(pseudo.domain(), source.codomain());
    assert!(std::ptr::eq(pseudo.provider(), provider.as_ref()));
    assert_moore_penrose!(&source, &pseudo, |actual: f64, expected: f64| (actual
        - expected)
        .abs());

    let complex = source.to_c64().scale(Complex64::new(1.0, 0.25));
    let complex_pseudo = complex.pinv(1e-12).unwrap();
    assert_moore_penrose!(
        &complex,
        &complex_pseudo,
        |actual: Complex64, expected: Complex64| (actual - expected).norm()
    );

    let tall: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&domain], [&codomain], |_, index| {
            [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]][index[0]][index[1]]
        })
        .unwrap();
    let tall_pseudo = tall.pinv(1e-12).unwrap();
    assert_moore_penrose!(&tall, &tall_pseudo, |actual: f64, expected: f64| (actual
        - expected)
        .abs());
    let complex_tall = tall.to_c64().scale(Complex64::new(1.0, 0.25));
    let complex_tall_pseudo = complex_tall.pinv(1e-12).unwrap();
    assert_moore_penrose!(
        &complex_tall,
        &complex_tall_pseudo,
        |actual: Complex64, expected: Complex64| { (actual - expected).norm() }
    );

    let lazy = source.adjoint().unwrap();
    let lazy_pseudo = lazy.pinv(1e-12).unwrap();
    for (actual, expected) in lazy_pseudo
        .data()
        .iter()
        .zip(pseudo.adjoint().unwrap().data())
    {
        assert!((actual - expected).abs() < 1e-10);
    }
}

fn assert_same_checked_generic_layout_and_close<R, D>(
    actual: &TensorMap<R, D>,
    expected: &TensorMap<R, D>,
    close: impl Fn(D, D) -> f64,
) where
    R: TypedSectorAdmission,
    D: tenet::typed::TensorScalar + fmt::Debug,
{
    assert_eq!(actual.block_count(), expected.block_count());
    for index in 0..actual.block_count() {
        let actual_block = actual.block(index).unwrap();
        let expected_block = expected.block(index).unwrap();
        assert_eq!(actual_block.key(), expected_block.key());
        assert_eq!(actual_block.shape(), expected_block.shape());
        assert_eq!(actual_block.strides(), expected_block.strides());
    }
    assert_eq!(actual.data().len(), expected.data().len());
    for (&actual, &expected) in actual.data().iter().zip(expected.data()) {
        assert!(
            close(actual, expected) < 1e-10,
            "{actual:?} != {expected:?}"
        );
    }
}

fn multiply_2x2<D: tenet::typed::TensorScalar>(
    left: [[D; 2]; 2],
    right: [[D; 2]; 2],
) -> [[D; 2]; 2] {
    std::array::from_fn(|row| {
        std::array::from_fn(|col| left[row][0] * right[0][col] + left[row][1] * right[1][col])
    })
}

#[test]
fn checked_generic_polar_matches_independent_real_and_complex_qh_oracles() {
    // What: both factor orders retain the source authority and match Q/H,
    // including conjugate-transpose arithmetic for a genuinely complex Q.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 2)]).unwrap();

    let h = [[2.0, 0.5], [0.5, 3.0]];
    let q = [[0.0, -1.0], [1.0, 0.0]];
    let left_data = multiply_2x2(q, h);
    let right_data = multiply_2x2(h, q);
    let q_tensor: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, ij| q[ij[0]][ij[1]]).unwrap();
    let h_tensor: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, ij| h[ij[0]][ij[1]]).unwrap();
    let left: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, ij| left_data[ij[0]][ij[1]])
            .unwrap();
    let right: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, ij| right_data[ij[0]][ij[1]])
            .unwrap();
    let (actual_q, actual_h) = left.left_polar().unwrap();
    assert!(std::ptr::eq(actual_q.provider(), provider.as_ref()));
    assert!(std::ptr::eq(actual_h.provider(), provider.as_ref()));
    assert!(actual_q.runtime().shares_state_with(left.runtime()));
    assert_same_checked_generic_layout_and_close(&actual_q, &q_tensor, |a, b| (a - b).abs());
    assert_same_checked_generic_layout_and_close(&actual_h, &h_tensor, |a, b| (a - b).abs());
    let (actual_h, actual_q) = right.right_polar().unwrap();
    assert_same_checked_generic_layout_and_close(&actual_q, &q_tensor, |a, b| (a - b).abs());
    assert_same_checked_generic_layout_and_close(&actual_h, &h_tensor, |a, b| (a - b).abs());

    let s = std::f64::consts::FRAC_1_SQRT_2;
    let cq = [
        [Complex64::new(s, 0.0), Complex64::new(0.0, s)],
        [Complex64::new(0.0, s), Complex64::new(s, 0.0)],
    ];
    let ch = [
        [Complex64::new(2.0, 0.0), Complex64::new(0.25, 0.5)],
        [Complex64::new(0.25, -0.5), Complex64::new(3.0, 0.0)],
    ];
    let complex_q: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, ij| cq[ij[0]][ij[1]]).unwrap();
    let complex_h: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, ij| ch[ij[0]][ij[1]]).unwrap();
    for (source, left) in [(multiply_2x2(cq, ch), true), (multiply_2x2(ch, cq), false)] {
        let source: TensorMap<_, Complex64> =
            TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, ij| source[ij[0]][ij[1]])
                .unwrap();
        let (actual_q, actual_h) = if left {
            source.left_polar().unwrap()
        } else {
            let (p, w) = source.right_polar().unwrap();
            (w, p)
        };
        assert_same_checked_generic_layout_and_close(&actual_q, &complex_q, |a, b| (a - b).norm());
        assert_same_checked_generic_layout_and_close(&actual_h, &complex_h, |a, b| (a - b).norm());
    }
}

#[test]
fn checked_generic_polar_direction_covers_rectangular_side_only_and_empty_inputs() {
    // What: complete dimensions, not populated blocks, select the requested
    // tall/wide contract before any dense factorization.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let tall = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 3)]).unwrap();
    let wide = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&tall], [&wide], |_, ij| {
            [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]][ij[0]][ij[1]]
        })
        .unwrap();
    assert!(source.left_polar().is_ok());
    assert!(matches!(
        source.right_polar(),
        Err(GenericTensorError::Plan(
            tenet::typed::CheckedGenericPlanError::Operation(_)
        ))
    ));
    let codomain_only =
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::Vacuum, 1), (Label::X, 1)])
            .unwrap();
    let domain_only =
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::Vacuum, 1)]).unwrap();
    let side_only: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&codomain_only], [&domain_only], |_, _| 0.0).unwrap();
    assert!(side_only.left_polar().is_ok());
    assert!(side_only.right_polar().is_err());
    let empty = GradedSpace::try_new_with_arc(Arc::clone(&provider), []).unwrap();
    let empty_map: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&empty], [&empty]).unwrap();
    assert!(empty_map.left_polar().unwrap().0.data().is_empty());
    assert!(empty_map.right_polar().unwrap().0.data().is_empty());
}

#[test]
fn checked_generic_polar_lazy_redirects_to_the_opposite_parent_operation() {
    // What: lazy adjoints reuse the parent's owned P, return owned W-adjoints,
    // and report the operation requested on the receiver.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, ij| {
            [
                [Complex64::new(2.0, 0.0), Complex64::new(0.5, 0.75)],
                [Complex64::new(-0.25, 0.5), Complex64::new(3.0, -0.25)],
            ][ij[0]][ij[1]]
        })
        .unwrap();
    let lazy = source.adjoint().unwrap();
    let (parent_p, parent_w) = source.right_polar().unwrap();
    let (actual_w, actual_p) = lazy.left_polar().unwrap();
    assert_same_checked_generic_layout_and_close(
        &actual_w,
        &parent_w.adjoint().unwrap(),
        |a, b| (a - b).norm(),
    );
    assert_same_checked_generic_layout_and_close(&actual_p, &parent_p, |a, b| (a - b).norm());
    let (parent_w, parent_p) = source.left_polar().unwrap();
    let (actual_p, actual_w) = lazy.right_polar().unwrap();
    assert_same_checked_generic_layout_and_close(&actual_p, &parent_p, |a, b| (a - b).norm());
    assert_same_checked_generic_layout_and_close(
        &actual_w,
        &parent_w.adjoint().unwrap(),
        |a, b| (a - b).norm(),
    );
    assert!(std::ptr::eq(actual_w.provider(), provider.as_ref()));
    assert!(actual_w.runtime().shares_state_with(source.runtime()));

    let tall = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 3)]).unwrap();
    let narrow = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let tall_source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&tall], [&narrow], |_, _| 1.0).unwrap();
    let error = tall_source.adjoint().unwrap().left_polar().unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Plan(tenet::typed::CheckedGenericPlanError::Operation(error))
            if matches!(
                error,
                tenet::operations::OperationError::InvalidArgument { message }
                    if message == "left_polar requires rows >= columns in every coupled-sector matrix"
            )
    ));
}

#[test]
fn checked_generic_polar_completes_rank_deficient_and_zero_sectors() {
    // What: compact-full SVD completion returns a full isometry/coisometry;
    // P remains Hermitian PSD for rank-deficient and zero matrices.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    for zero in [false, true] {
        let source: TensorMap<_, Complex64> =
            TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, ij| {
                if zero {
                    Complex64::new(0.0, 0.0)
                } else {
                    let row = [Complex64::new(1.0, 1.0), Complex64::new(2.0, -0.5)];
                    let col = [Complex64::new(0.5, -1.0), Complex64::new(-1.5, 2.0)];
                    row[ij[0]] * col[ij[1]].conj()
                }
            })
            .unwrap();
        for left in [true, false] {
            let (w, p) = if left {
                source.left_polar().unwrap()
            } else {
                let (p, w) = source.right_polar().unwrap();
                (w, p)
            };
            let rebuilt = if left {
                w.compose(&p).unwrap()
            } else {
                p.compose(&w).unwrap()
            };
            for (&actual, &expected) in rebuilt.data().iter().zip(source.data()) {
                assert!((actual - expected).norm() < 1e-9);
            }
            for col in 0..2 {
                for row in 0..2 {
                    let gram = (0..2).fold(Complex64::new(0.0, 0.0), |sum, inner| {
                        if left {
                            sum + w.data()[inner + 2 * row].conj() * w.data()[inner + 2 * col]
                        } else {
                            sum + w.data()[row + 2 * inner] * w.data()[col + 2 * inner].conj()
                        }
                    });
                    assert!((gram - Complex64::new(f64::from(row == col), 0.0)).norm() < 1e-10);
                    assert!(
                        (p.data()[row + 2 * col] - p.data()[col + 2 * row].conj()).norm() < 1e-10
                    );
                }
            }
            assert!(p
                .eigh_vals()
                .unwrap()
                .iter()
                .flat_map(|entry| &entry.values)
                .all(|&value| value >= -1e-10));
        }
    }
}

struct PinvFaultExecutor {
    inner: DefaultDenseExecutor,
    svd_calls: Arc<AtomicUsize>,
    gemm_calls: Arc<AtomicUsize>,
    fail_svd: Option<usize>,
    fail_gemm: Option<usize>,
}

impl DenseExecutor for PinvFaultExecutor {
    fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.svd(input)
    }

    fn svd_into(
        &mut self,
        input: DenseRead<'_>,
        u: DenseWrite<'_>,
        s: DenseWrite<'_>,
        vt: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        let call = self.svd_calls.fetch_add(1, Ordering::Relaxed) + 1;
        if self.fail_svd == Some(call) {
            return Err(DenseError::Backend {
                backend: DenseBackend::Tenferro,
                op: "svd_into",
                message: "injected pinv SVD failure".to_string(),
            });
        }
        self.inner.svd_into(input, u, s, vt)
    }

    fn qr(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.qr(input)
    }

    fn eigh(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.eigh(input)
    }

    fn dot_general_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        config: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        let call = self.gemm_calls.fetch_add(1, Ordering::Relaxed) + 1;
        if self.fail_gemm == Some(call) {
            return Err(DenseError::Backend {
                backend: DenseBackend::Tenferro,
                op: "dot_general_into",
                message: "injected pinv GEMM failure".to_string(),
            });
        }
        self.inner.dot_general_into(output, lhs, rhs, config)
    }
}

#[test]
fn checked_generic_polar_stages_svd_and_both_gemms_without_publication() {
    // What: every SVD completes before W/P GEMMs, and either GEMM failure
    // returns no factor while preserving the source and provider authority.
    for (fail_svd, fail_gemm, expected_svd, expected_gemm) in [
        (None, None, 2, 4),
        (Some(1), None, 1, 0),
        (Some(2), None, 2, 0),
        (None, Some(1), 2, 1),
        (None, Some(2), 2, 2),
        (None, Some(4), 2, 4),
    ] {
        let svd_calls = Arc::new(AtomicUsize::new(0));
        let gemm_calls = Arc::new(AtomicUsize::new(0));
        let runtime = Runtime::builder()
            .dense_threads(1)
            .with_dense_executor(Box::new(PinvFaultExecutor {
                inner: DefaultDenseExecutor::new(),
                svd_calls: Arc::clone(&svd_calls),
                gemm_calls: Arc::clone(&gemm_calls),
                fail_svd,
                fail_gemm,
            }))
            .build()
            .unwrap();
        let provider = Arc::new(CheckedOnlyToy::new(0));
        let bond = GradedSpace::try_new_with_arc(
            Arc::clone(&provider),
            [(Label::Vacuum, 1), (Label::X, 1)],
        )
        .unwrap();
        let source: TensorMap<_, f64> =
            TensorMap::from_block_fn(&runtime, [&bond], [&bond], |trees, _| {
                if trees.coupled() == &Label::Vacuum {
                    2.0
                } else {
                    3.0
                }
            })
            .unwrap();
        let before = source.data().to_vec();
        let result = source.left_polar();
        if fail_svd.is_some() || fail_gemm.is_some() {
            assert!(matches!(
                result,
                Err(GenericTensorError::Plan(
                    tenet::typed::CheckedGenericPlanError::Operation(_)
                ))
            ));
        } else {
            let (w, p) = result.unwrap();
            assert!(std::ptr::eq(w.provider(), provider.as_ref()));
            assert!(std::ptr::eq(p.provider(), provider.as_ref()));
        }
        assert_eq!(svd_calls.load(Ordering::Relaxed), expected_svd);
        assert_eq!(gemm_calls.load(Ordering::Relaxed), expected_gemm);
        assert_eq!(source.data(), before.as_slice());
    }
}

#[test]
fn checked_generic_polar_provider_error_precedes_dense_work() {
    // What: complete codomain dimensions are queried before domain,
    // direction, admission, and dense work.
    let svd_calls = Arc::new(AtomicUsize::new(0));
    let gemm_calls = Arc::new(AtomicUsize::new(0));
    let runtime = Runtime::builder()
        .dense_threads(1)
        .with_dense_executor(Box::new(PinvFaultExecutor {
            inner: DefaultDenseExecutor::new(),
            svd_calls: Arc::clone(&svd_calls),
            gemm_calls: Arc::clone(&gemm_calls),
            fail_svd: None,
            fail_gemm: None,
        }))
        .build()
        .unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&leg], [&leg]).unwrap();
    let before = source.data().to_vec();
    provider.fail_algebra.store(true, Ordering::Relaxed);
    assert!(matches!(
        source.left_polar(),
        Err(GenericTensorError::Plan(
            tenet::typed::CheckedGenericPlanError::Provider(ToyError::Algebra)
        ))
    ));
    assert_eq!(svd_calls.load(Ordering::Relaxed), 0);
    assert_eq!(gemm_calls.load(Ordering::Relaxed), 0);
    assert_eq!(source.data(), before);
}

#[test]
fn checked_generic_pinv_stages_svd_and_gemm_failures_without_publication() {
    for (fail_svd, fail_gemm, expected_svd, expected_gemm) in [
        (None, None, 2, 2),
        (Some(1), None, 1, 0),
        (Some(2), None, 2, 0),
        (None, Some(2), 2, 2),
    ] {
        let svd_calls = Arc::new(AtomicUsize::new(0));
        let gemm_calls = Arc::new(AtomicUsize::new(0));
        let runtime = Runtime::builder()
            .dense_threads(1)
            .with_dense_executor(Box::new(PinvFaultExecutor {
                inner: DefaultDenseExecutor::new(),
                svd_calls: Arc::clone(&svd_calls),
                gemm_calls: Arc::clone(&gemm_calls),
                fail_svd,
                fail_gemm,
            }))
            .build()
            .unwrap();
        let provider = Arc::new(CheckedOnlyToy::new(0));
        let bond = GradedSpace::try_new_with_arc(
            Arc::clone(&provider),
            [(Label::Vacuum, 1), (Label::X, 1)],
        )
        .unwrap();
        let source: TensorMap<_, f64> =
            TensorMap::from_block_fn(&runtime, [&bond], [&bond], |trees, _| {
                if trees.coupled() == &Label::Vacuum {
                    2.0
                } else {
                    3.0
                }
            })
            .unwrap();
        let before = source.data().to_vec();
        let result = source.pinv(0.0);
        if fail_svd.is_some() || fail_gemm.is_some() {
            assert!(matches!(
                result,
                Err(GenericTensorError::Facade(tenet::typed::Error::Operation(
                    _
                )))
            ));
        } else {
            assert!(result.is_ok());
        }
        assert_eq!(svd_calls.load(Ordering::Relaxed), expected_svd);
        assert_eq!(gemm_calls.load(Ordering::Relaxed), expected_gemm);
        assert_eq!(source.data(), before.as_slice());
    }
}

#[test]
fn checked_generic_pinv_uses_a_strict_global_cutoff() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let bond =
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::Vacuum, 1), (Label::X, 1)])
            .unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&bond], [&bond], |trees, _| {
            if trees.coupled() == &Label::Vacuum {
                4.0
            } else {
                2.0
            }
        })
        .unwrap();
    let pseudo = source.pinv(0.5).unwrap();
    assert_eq!(pseudo.data(), &[0.25, 0.0]);
}

#[test]
fn checked_generic_null_spaces_cover_rank_cutoff_zero_disjoint_and_side_only_sectors() {
    // What: Generic null spaces use the documented numerical rank, keep full
    // zero/side-only directions, drop full-rank sectors, and retain authority.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let x = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let tolerance = f64::EPSILON * 2.0;
    for (small, nullity) in [(0.5 * tolerance, 1), (2.0 * tolerance, 0)] {
        let source: TensorMap<_, f64> =
            TensorMap::from_block_fn(&runtime, [&x], [&x], |_, index| match index {
                [0, 0] => 1.0,
                [1, 1] => small,
                _ => 0.0,
            })
            .unwrap();
        for null in [source.left_null().unwrap(), source.right_null().unwrap()] {
            assert!(std::ptr::eq(null.provider(), provider.as_ref()));
            if nullity == 0 {
                assert!(null.data().is_empty());
            } else {
                assert_eq!(null.data().len(), 2);
            }
        }
    }

    let zero: TensorMap<_, Complex64> = TensorMap::zeros(&runtime, [&x], [&x]).unwrap();
    assert_eq!(zero.left_null().unwrap().data().len(), 4);
    assert_eq!(zero.right_null().unwrap().data().len(), 4);

    let vacuum =
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::Vacuum, 3)]).unwrap();
    let disjoint: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&x], [&vacuum]).unwrap();
    let left = disjoint.left_null().unwrap();
    let right = disjoint.right_null().unwrap();
    assert_eq!(left.data(), &[1.0, 0.0, 0.0, 1.0]);
    assert_eq!(right.data().len(), 9);
    for column in 0..3 {
        for row in 0..3 {
            assert_eq!(right.data()[row + 3 * column], f64::from(row == column));
        }
    }

    let shared =
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::Vacuum, 1), (Label::X, 2)])
            .unwrap();
    let unit = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::Vacuum, 1)]).unwrap();
    let codomain_side: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&shared], [&unit], |trees, index| {
            f64::from(trees.coupled() == &Label::Vacuum && index[0] == index[1])
        })
        .unwrap();
    assert_eq!(codomain_side.left_null().unwrap().data().len(), 4);
    assert!(codomain_side.right_null().unwrap().data().is_empty());
}

#[test]
fn checked_generic_null_dense_failure_is_typed_and_nonpublishing() {
    // What: a later sector SVD failure crosses the public Generic facade as a
    // typed plan error without changing the source or returning a partial null.
    let svd_calls = Arc::new(AtomicUsize::new(0));
    let runtime = Runtime::builder()
        .dense_threads(1)
        .with_dense_executor(Box::new(PinvFaultExecutor {
            inner: DefaultDenseExecutor::new(),
            svd_calls: Arc::clone(&svd_calls),
            gemm_calls: Arc::new(AtomicUsize::new(0)),
            fail_svd: Some(2),
            fail_gemm: None,
        }))
        .build()
        .unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg =
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::Vacuum, 1), (Label::X, 1)])
            .unwrap();
    let source: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&leg], [&leg]).unwrap();
    let before = source.data().to_vec();
    assert!(matches!(
        source.left_null(),
        Err(GenericTensorError::Plan(
            tenet::typed::CheckedGenericPlanError::Operation(_)
        ))
    ));
    assert_eq!(svd_calls.load(Ordering::Relaxed), 2);
    assert_eq!(source.data(), before);
    assert!(std::ptr::eq(source.provider(), provider.as_ref()));
}

#[cfg(feature = "racah-generated")]
fn assert_sun_checked_generic_null_projectors<D>(
    n: usize,
    label: Vec<i64>,
    u: [D; 2],
    v: [D; 2],
    u_norm_squared: f64,
    v_norm_squared: f64,
    adjoint: impl Fn(D) -> D,
    close: impl Fn(D, D) -> f64,
) where
    D: tenet::typed::TensorScalar + fmt::Debug + PartialEq,
{
    use tenet::typed::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(n).unwrap());
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(label, 2)]).unwrap();
    let source: TensorMap<_, D> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, index| {
            let row = index[0] + 2 * index[1];
            let column = index[2] + 2 * index[3];
            if row == column {
                u[trees.codomain_vertices()[0].get() - 1]
                    * adjoint(v[trees.domain_vertices()[0].get() - 1])
            } else {
                D::from_real(0.0)
            }
        })
        .unwrap();
    assert!((0..source.block_count()).any(|index| {
        let trees = source.block_fusion_trees(index).unwrap();
        trees.codomain_vertices()[0].get() == 2
            && trees.domain_vertices()[0].get() == 1
            && source.data()[source.block(index).unwrap().offset()] != D::from_real(0.0)
    }));
    let outer_multiplicity_sectors = (0..source.block_count())
        .filter_map(|index| {
            let trees = source.block_fusion_trees(index).unwrap();
            trees
                .codomain_vertices()
                .iter()
                .chain(trees.domain_vertices())
                .any(|vertex| vertex.get() > 1)
                .then(|| trees.coupled().clone())
        })
        .collect::<Vec<_>>();

    let left = source.left_null().unwrap();
    let right = source.right_null().unwrap();
    assert!(std::ptr::eq(left.provider(), provider.as_ref()));
    assert!(std::ptr::eq(right.provider(), provider.as_ref()));
    let left_adjoint = left.adjoint().unwrap();
    let left_adjoint = left_adjoint
        .add(&left_adjoint, D::from_real(1.0), D::from_real(0.0))
        .unwrap();
    let right_adjoint = right.adjoint().unwrap();
    let right_adjoint = right_adjoint
        .add(&right_adjoint, D::from_real(1.0), D::from_real(0.0))
        .unwrap();
    let left_projector = left.compose(&left_adjoint).unwrap();
    let right_projector = right_adjoint.compose(&right).unwrap();
    let expected_left: TensorMap<_, D> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, index| {
            let row = index[0] + 2 * index[1];
            let column = index[2] + 2 * index[3];
            if row != column || !outer_multiplicity_sectors.contains(trees.coupled()) {
                return D::from_real(0.0);
            }
            let i = trees.codomain_vertices()[0].get() - 1;
            let j = trees.domain_vertices()[0].get() - 1;
            D::from_real(f64::from(i == j))
                + D::from_real(-1.0 / u_norm_squared) * u[i] * adjoint(u[j])
        })
        .unwrap();
    let expected_right: TensorMap<_, D> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, index| {
            let row = index[0] + 2 * index[1];
            let column = index[2] + 2 * index[3];
            if row != column || !outer_multiplicity_sectors.contains(trees.coupled()) {
                return D::from_real(0.0);
            }
            let i = trees.codomain_vertices()[0].get() - 1;
            let j = trees.domain_vertices()[0].get() - 1;
            D::from_real(f64::from(i == j))
                + D::from_real(-1.0 / v_norm_squared) * v[i] * adjoint(v[j])
        })
        .unwrap();
    for (name, actual, expected) in [
        ("left", &left_projector, &expected_left),
        ("right", &right_projector, &expected_right),
    ] {
        assert_eq!(actual.block_count(), expected.block_count());
        for index in 0..actual.block_count() {
            let actual = actual.block(index).unwrap();
            let expected = expected.block(index).unwrap();
            assert_eq!(actual.key(), expected.key());
            assert_eq!(actual.shape(), expected.shape());
            assert_eq!(actual.strides(), expected.strides());
        }
        for (index, (&actual, &expected)) in actual.data().iter().zip(expected.data()).enumerate() {
            assert!(
                close(actual, expected) < 1e-9,
                "{name} projector mismatch at raw {index}: {actual:?} != {expected:?}"
            );
        }
    }

    for value in left_adjoint.compose(&source).unwrap().data() {
        assert!(close(*value, D::from_real(0.0)) < 1e-9);
    }
    for value in source.compose(&right_adjoint).unwrap().data() {
        assert!(close(*value, D::from_real(0.0)) < 1e-9);
    }
    for gram in [
        left_adjoint.compose(&left).unwrap(),
        right.compose(&right_adjoint).unwrap(),
    ] {
        for block_index in 0..gram.block_count() {
            let block = gram.block(block_index).unwrap();
            for column in 0..block.shape()[1] {
                for row in 0..block.shape()[0] {
                    let expected = D::from_real(f64::from(row == column));
                    let actual = gram.data()
                        [block.offset() + row * block.strides()[0] + column * block.strides()[1]];
                    assert!(close(actual, expected) < 1e-9);
                }
            }
        }
    }

    let lazy = source.adjoint().unwrap();
    let lazy_left = lazy.left_null().unwrap();
    let expected_lazy_left = right_adjoint;
    assert!(std::ptr::eq(lazy_left.provider(), provider.as_ref()));
    assert_eq!(lazy_left.data(), expected_lazy_left.data());
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_checked_generic_null_projectors_resolve_cross_mu_for_both_dtypes() {
    for (n, label) in [(3, vec![1, 1]), (4, vec![1, 0, 1])] {
        assert_sun_checked_generic_null_projectors::<f64>(
            n,
            label.clone(),
            [1.0, 2.0],
            [1.0, -1.0],
            5.0,
            2.0,
            |value| value,
            |actual, expected| (actual - expected).abs(),
        );
        assert_sun_checked_generic_null_projectors::<Complex64>(
            n,
            label,
            [Complex64::new(1.0, 1.0), Complex64::new(2.0, -0.5)],
            [Complex64::new(0.5, -1.0), Complex64::new(-1.5, 2.0)],
            6.25,
            7.5,
            |value| value.conj(),
            |actual, expected| (actual - expected).norm(),
        );
    }
}

#[cfg(feature = "racah-generated")]
fn assert_sun_checked_generic_pinv<D>(
    n: usize,
    label: Vec<i64>,
    off_diagonal: D,
    close: impl Fn(D, D) -> f64,
) where
    D: tenet::typed::TensorScalar + fmt::Debug + PartialEq,
{
    use tenet::typed::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(n).unwrap());
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(label, 2)]).unwrap();
    let source: TensorMap<_, D> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, index| {
            let row = index[0] + 2 * index[1];
            let column = index[2] + 2 * index[3];
            if trees.codomain_vertices() == trees.domain_vertices() && row == column {
                D::from_real(2.0)
            } else if trees.codomain_vertices()[0].get() == 2
                && trees.domain_vertices()[0].get() == 1
                && row == column
            {
                off_diagonal
            } else {
                D::from_real(0.0)
            }
        })
        .unwrap();
    assert!((0..source.block_count()).any(|index| {
        source
            .block_fusion_trees(index)
            .unwrap()
            .codomain_vertices()
            .iter()
            .chain(source.block_fusion_trees(index).unwrap().domain_vertices())
            .any(|vertex| vertex.get() > 1)
    }));
    let pseudo = source.pinv(1e-12).unwrap();
    assert!(std::ptr::eq(pseudo.provider(), provider.as_ref()));
    assert_eq!(pseudo.codomain(), source.domain());
    assert_eq!(pseudo.domain(), source.codomain());
    let expected: TensorMap<_, D> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, index| {
            let row = index[0] + 2 * index[1];
            let column = index[2] + 2 * index[3];
            if trees.codomain_vertices() == trees.domain_vertices() && row == column {
                D::from_real(0.5)
            } else if trees.codomain_vertices()[0].get() == 2
                && trees.domain_vertices()[0].get() == 1
                && row == column
            {
                D::from_real(-0.25) * off_diagonal
            } else {
                D::from_real(0.0)
            }
        })
        .unwrap();
    for index in 0..pseudo.block_count() {
        let actual = pseudo.block(index).unwrap();
        let expected_block = expected.block(index).unwrap();
        assert_eq!(actual.key(), expected_block.key());
        assert_eq!(actual.shape(), expected_block.shape());
        assert_eq!(actual.strides(), expected_block.strides());
    }
    for (actual, expected) in pseudo.data().iter().zip(expected.data()) {
        assert!(close(*actual, *expected) < 1e-9);
    }
    assert!((0..source.block_count()).any(|index| {
        let trees = source.block_fusion_trees(index).unwrap();
        trees.codomain_vertices()[0].get() == 2
            && trees.domain_vertices()[0].get() == 1
            && source.block(index).unwrap().shape() == [2, 2, 2, 2]
            && source.data()[source.block(index).unwrap().offset()] == off_diagonal
    }));
    let aa_plus = source.compose(&pseudo).unwrap();
    let a_plus_a = pseudo.compose(&source).unwrap();
    for (actual, expected) in aa_plus
        .compose(&source)
        .unwrap()
        .data()
        .iter()
        .zip(source.data())
    {
        assert!(close(*actual, *expected) < 1e-9);
    }
    for (actual, expected) in a_plus_a
        .compose(&pseudo)
        .unwrap()
        .data()
        .iter()
        .zip(pseudo.data())
    {
        assert!(close(*actual, *expected) < 1e-9);
    }
    for (actual, expected) in aa_plus.adjoint().unwrap().data().iter().zip(aa_plus.data()) {
        assert!(close(*actual, *expected) < 1e-9);
    }
    for (actual, expected) in a_plus_a
        .adjoint()
        .unwrap()
        .data()
        .iter()
        .zip(a_plus_a.data())
    {
        assert!(close(*actual, *expected) < 1e-9);
    }
    let lazy = source.adjoint().unwrap();
    let lazy_pseudo = lazy.pinv(1e-12).unwrap();
    for (actual, expected) in lazy_pseudo
        .data()
        .iter()
        .zip(pseudo.adjoint().unwrap().data())
    {
        assert!(close(*actual, *expected) < 1e-9);
    }
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_checked_generic_pinv_cross_mu_full_keys_for_both_dtypes() {
    for (n, label) in [(3, vec![1, 1]), (4, vec![1, 0, 1])] {
        assert_sun_checked_generic_pinv::<f64>(n, label.clone(), 1.0, |actual, expected| {
            (actual - expected).abs()
        });
        assert_sun_checked_generic_pinv::<Complex64>(
            n,
            label,
            Complex64::new(1.0, 0.5),
            |actual, expected| (actual - expected).norm(),
        );
    }
}

#[cfg(feature = "racah-generated")]
fn assert_sun_checked_generic_polar_qh<D>(
    n: usize,
    label: Vec<i64>,
    q: [[D; 2]; 2],
    h: [[D; 2]; 2],
    close: impl Fn(D, D) -> f64 + Copy,
) where
    D: tenet::typed::TensorScalar + fmt::Debug + PartialEq,
{
    use tenet::typed::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(n).unwrap());
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(label.clone(), 2)]).unwrap();
    let build = |matrix: [[D; 2]; 2], fallback: D| {
        let cross_sector = label.clone();
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg, &leg], move |trees, index| {
            let row = index[0] + 2 * index[1];
            let col = index[2] + 2 * index[3];
            if row != col {
                return D::from_real(0.0);
            }
            if trees.coupled() == &cross_sector {
                matrix[trees.codomain_vertices()[0].get() - 1][trees.domain_vertices()[0].get() - 1]
            } else if trees.codomain_vertices() == trees.domain_vertices() {
                fallback
            } else {
                D::from_real(0.0)
            }
        })
        .unwrap()
    };
    let expected_q = build(q, D::from_real(1.0));
    let expected_h = build(h, D::from_real(2.0));
    let mut saw_cross_mu = false;
    for (source_matrix, left) in [(multiply_2x2(q, h), true), (multiply_2x2(h, q), false)] {
        let source = build(source_matrix, D::from_real(2.0));
        saw_cross_mu |= (0..source.block_count()).any(|index| {
            let trees = source.block_fusion_trees(index).unwrap();
            trees.coupled() == &label
                && trees.codomain_vertices()[0].get() == 2
                && trees.domain_vertices()[0].get() == 1
        });
        let (actual_q, actual_h) = if left {
            source.left_polar().unwrap()
        } else {
            let (p, w) = source.right_polar().unwrap();
            (w, p)
        };
        assert!(std::ptr::eq(actual_q.provider(), provider.as_ref()));
        assert!(std::ptr::eq(actual_h.provider(), provider.as_ref()));
        assert_eq!(actual_q.codomain(), source.codomain());
        assert_eq!(actual_q.domain(), source.domain());
        assert_same_checked_generic_layout_and_close(&actual_q, &expected_q, close);
        assert_same_checked_generic_layout_and_close(&actual_h, &expected_h, close);
        let rebuilt = if left {
            actual_q.compose(&actual_h).unwrap()
        } else {
            actual_h.compose(&actual_q).unwrap()
        };
        assert_same_checked_generic_layout_and_close(&rebuilt, &source, close);
    }
    assert!(
        saw_cross_mu,
        "SU(N) polar fixture must carry a cross-mu full key"
    );
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_checked_generic_polar_cross_mu_qh_oracles_for_both_dtypes() {
    let s = std::f64::consts::FRAC_1_SQRT_2;
    for (n, label) in [(3, vec![1, 1]), (4, vec![1, 0, 1])] {
        assert_sun_checked_generic_polar_qh::<f64>(
            n,
            label.clone(),
            [[0.0, -1.0], [1.0, 0.0]],
            [[2.0, 0.5], [0.5, 3.0]],
            |actual, expected| (actual - expected).abs(),
        );
        assert_sun_checked_generic_polar_qh::<Complex64>(
            n,
            label,
            [
                [Complex64::new(s, 0.0), Complex64::new(0.0, s)],
                [Complex64::new(0.0, s), Complex64::new(s, 0.0)],
            ],
            [
                [Complex64::new(2.0, 0.0), Complex64::new(0.25, 0.5)],
                [Complex64::new(0.25, -0.5), Complex64::new(3.0, 0.0)],
            ],
            |actual, expected| (actual - expected).norm(),
        );
    }
}

#[cfg(feature = "racah-generated")]
fn assert_sun_checked_generic_solve_right<D>(
    n: usize,
    label: Vec<i64>,
    off_diagonal: D,
    close: impl Fn(D, D) -> f64,
) where
    D: tenet::typed::TensorScalar + fmt::Debug + PartialEq,
{
    use tenet::typed::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let receiver_provider = Arc::new(SUNFusionRule::new(n).unwrap());
    let divisor_provider = Arc::new(SUNFusionRule::new(n).unwrap());
    let cross_sector = label.clone();
    let receiver_leg =
        GradedSpace::try_new_with_arc(Arc::clone(&receiver_provider), [(label.clone(), 2)])
            .unwrap();
    let divisor_leg =
        GradedSpace::try_new_with_arc(Arc::clone(&divisor_provider), [(label, 2)]).unwrap();
    let receiver_off_diagonal = D::from_real(0.75);
    let receiver: TensorMap<_, D> = TensorMap::from_block_fn(
        &runtime,
        [&receiver_leg, &receiver_leg],
        [&receiver_leg, &receiver_leg],
        |trees, index| {
            let row = index[0] + 2 * index[1];
            let column = index[2] + 2 * index[3];
            if trees.codomain_vertices() == trees.domain_vertices() && row == column {
                D::from_real(3.0)
            } else if trees.codomain_vertices()[0].get() == 1
                && trees.domain_vertices()[0].get() == 2
                && row == column
            {
                receiver_off_diagonal
            } else {
                D::from_real(0.0)
            }
        },
    )
    .unwrap();
    let divisor: TensorMap<_, D> = TensorMap::from_block_fn(
        &runtime,
        [&divisor_leg, &divisor_leg],
        [&divisor_leg, &divisor_leg],
        |trees, index| {
            let row = index[0] + 2 * index[1];
            let column = index[2] + 2 * index[3];
            if trees.codomain_vertices() == trees.domain_vertices() && row == column {
                D::from_real(2.0)
            } else if trees.codomain_vertices()[0].get() == 2
                && trees.domain_vertices()[0].get() == 1
                && row == column
            {
                off_diagonal
            } else {
                D::from_real(0.0)
            }
        },
    )
    .unwrap();
    let ab = receiver.compose(&divisor).unwrap();
    let ba = divisor.compose(&receiver).unwrap();
    assert!(ab
        .data()
        .iter()
        .zip(ba.data())
        .any(|(&left, &right)| close(left, right) > 1e-9));

    let solution = receiver.solve_right(&divisor).unwrap();
    assert!(std::ptr::eq(
        solution.provider(),
        receiver_provider.as_ref()
    ));
    assert!(!std::ptr::eq(
        solution.provider(),
        divisor_provider.as_ref()
    ));
    assert_eq!(solution.codomain(), receiver.codomain());
    assert_eq!(solution.domain(), divisor.codomain());
    // What: for `M=|1><2|`, `N=|2><1|`, and `MN=P1`,
    // `(3I+mM)(2I+nN)^-1 = 3/2 I - 3n/4 N + m/2 M - mn/4 P1`.
    let expected: TensorMap<_, D> = TensorMap::from_block_fn(
        &runtime,
        [&receiver_leg, &receiver_leg],
        [&receiver_leg, &receiver_leg],
        |trees, index| {
            let row = index[0] + 2 * index[1];
            let column = index[2] + 2 * index[3];
            if trees.codomain_vertices() == trees.domain_vertices() && row == column {
                let correction = if trees.coupled() == &cross_sector
                    && trees.codomain_vertices()[0].get() == 1
                {
                    D::from_real(-0.25) * receiver_off_diagonal * off_diagonal
                } else {
                    D::from_real(0.0)
                };
                D::from_real(1.5) + correction
            } else if trees.codomain_vertices()[0].get() == 2
                && trees.domain_vertices()[0].get() == 1
                && row == column
            {
                D::from_real(-0.75) * off_diagonal
            } else if trees.codomain_vertices()[0].get() == 1
                && trees.domain_vertices()[0].get() == 2
                && row == column
            {
                D::from_real(0.5) * receiver_off_diagonal
            } else {
                D::from_real(0.0)
            }
        },
    )
    .unwrap();
    for index in 0..solution.block_count() {
        let actual = solution.block(index).unwrap();
        let expected_block = expected.block(index).unwrap();
        assert_eq!(actual.key(), expected_block.key());
        assert_eq!(actual.shape(), expected_block.shape());
        assert_eq!(actual.strides(), expected_block.strides());
    }
    for (index, (&actual, &expected)) in solution.data().iter().zip(expected.data()).enumerate() {
        assert!(
            close(actual, expected) < 1e-9,
            "payload {index}: actual={actual:?}, expected={expected:?}"
        );
    }
    assert!(solution
        .compose(&divisor)
        .unwrap()
        .data()
        .iter()
        .zip(receiver.data())
        .all(|(&actual, &expected)| close(actual, expected) < 1e-9));
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_checked_generic_solve_right_cross_mu_for_both_dtypes_and_receiver_arcs() {
    // What: noncommuting cross-multiplicity matrices distinguish `A / B`
    // from left solve and exercise both payload types and receiver Arc authority.
    for (n, label) in [(3, vec![1, 1]), (4, vec![1, 0, 1])] {
        assert_sun_checked_generic_solve_right::<f64>(n, label.clone(), 1.0, |actual, expected| {
            (actual - expected).abs()
        });
        assert_sun_checked_generic_solve_right::<Complex64>(
            n,
            label,
            Complex64::new(1.0, 0.5),
            |actual, expected| (actual - expected).norm(),
        );
    }
}

struct SolveCallSpy {
    inner: DefaultDenseExecutor,
    calls: Arc<AtomicUsize>,
}

impl DenseExecutor for SolveCallSpy {
    fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.svd(input)
    }

    fn qr(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.qr(input)
    }

    fn eigh(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.eigh(input)
    }

    fn solve_into(
        &mut self,
        a: DenseRead<'_>,
        b: DenseRead<'_>,
        x: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.solve_into(a, b, x)
    }

    fn dot_general_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        config: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        self.inner.dot_general_into(output, lhs, rhs, config)
    }
}

#[test]
fn checked_generic_inv_singular_early_and_late_sectors_preserve_source() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let bond =
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::Vacuum, 1), (Label::X, 1)])
            .unwrap();
    for target in [Label::Vacuum, Label::X] {
        let source: TensorMap<_, f64> =
            TensorMap::from_block_fn(&runtime, [&bond], [&bond], |trees, _| {
                if trees.coupled() == &target {
                    0.0
                } else {
                    1.0
                }
            })
            .unwrap();
        let labels = (0..source.block_count())
            .map(|index| source.block_fusion_trees(index).unwrap().coupled().clone())
            .collect::<Vec<_>>();
        assert_eq!(labels, [Label::Vacuum, Label::X]);
        let before = source.data().to_vec();
        assert!(matches!(
            source.inv(),
            Err(GenericTensorError::Facade(tenet::typed::Error::Operation(
                _
            )))
        ));
        assert_eq!(source.data(), before.as_slice());
    }
}

#[test]
fn checked_generic_left_solve_accepts_distinct_provider_arcs_and_rectangular_rhs() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let lhs_provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let rhs_provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let lhs_codomain =
        GradedSpace::try_new_with_arc(Arc::clone(&lhs_provider), [(Label::X, 2)]).unwrap();
    let lhs_domain_x =
        GradedSpace::try_new_with_arc(Arc::clone(&lhs_provider), [(Label::X, 2)]).unwrap();
    let lhs_domain_unit =
        GradedSpace::try_new_with_arc(Arc::clone(&lhs_provider), [(Label::Vacuum, 1)]).unwrap();
    let rhs_codomain =
        GradedSpace::try_new_with_arc(Arc::clone(&rhs_provider), [(Label::X, 2)]).unwrap();
    let rhs_domain = GradedSpace::try_new_with_arc(rhs_provider, [(Label::X, 3)]).unwrap();
    let divisor: TensorMap<_, f64> = TensorMap::from_block_fn(
        &runtime,
        [&lhs_codomain],
        [&lhs_domain_x, &lhs_domain_unit],
        |_, indices| {
            if indices[0] == indices[1] {
                2.0 + indices[0] as f64
            } else {
                0.0
            }
        },
    )
    .unwrap();
    let rhs: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&rhs_codomain], [&rhs_domain], |_, indices| {
            (indices[0] + 2 * indices[1] + 1) as f64
        })
        .unwrap();

    let solution = divisor.solve(&rhs).unwrap();
    assert!(std::ptr::eq(solution.provider(), lhs_provider.as_ref()));
    assert_eq!(solution.codomain(), divisor.domain());
    assert_eq!(solution.domain(), rhs.domain());
    assert!(divisor
        .compose(&solution)
        .unwrap()
        .data()
        .iter()
        .zip(rhs.data())
        .all(|(actual, expected)| (*actual - *expected).abs() < 1e-11));

    let complex_divisor = divisor.to_c64();
    let complex_rhs = rhs.to_c64().scale(Complex64::new(1.0, 0.25));
    let complex_solution = complex_divisor.solve(&complex_rhs).unwrap();
    assert!(std::ptr::eq(
        complex_solution.provider(),
        lhs_provider.as_ref()
    ));
    assert!(complex_divisor
        .compose(&complex_solution)
        .unwrap()
        .data()
        .iter()
        .zip(complex_rhs.data())
        .all(|(actual, expected)| (*actual - *expected).norm() < 1e-11));
}

#[test]
fn checked_generic_left_solve_preflight_failures_are_nonpublishing() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let x = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let narrow = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let lhs: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&x], [&narrow], |_, _| 1.0).unwrap();
    let rhs: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&x], [&x], |_, _| 1.0).unwrap();
    let before = lhs.data().to_vec();
    assert!(matches!(
        lhs.solve(&rhs),
        Err(GenericTensorError::Facade(tenet::typed::Error::Operation(
            _
        )))
    ));
    assert_eq!(lhs.data(), before.as_slice());

    let other_runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let runtime_rhs: TensorMap<_, f64> =
        TensorMap::from_block_fn(&other_runtime, [&x], [&x], |_, _| 1.0).unwrap();
    reset_provider_queries(&provider);
    assert!(matches!(
        lhs.solve(&runtime_rhs),
        Err(GenericTensorError::Facade(
            tenet::typed::Error::RuntimeMismatch
        ))
    ));
    assert_no_provider_queries(&provider);

    let foreign_provider = Arc::new(CheckedOnlyToy::new_product_probe(1));
    let foreign_x =
        GradedSpace::try_new_with_arc(Arc::clone(&foreign_provider), [(Label::X, 2)]).unwrap();
    let foreign_rhs: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&foreign_x], [&foreign_x], |_, _| 1.0).unwrap();
    reset_provider_queries(&provider);
    reset_provider_queries(&foreign_provider);
    assert!(matches!(
        lhs.solve(&foreign_rhs),
        Err(GenericTensorError::Facade(
            tenet::typed::Error::RuleMismatch
        ))
    ));
    assert_no_provider_queries(&provider);
    assert_no_provider_queries(&foreign_provider);

    let wrong_codomain =
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::Vacuum, 1)]).unwrap();
    let codomain_rhs: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&wrong_codomain], [&x], |_, _| 1.0).unwrap();
    reset_provider_queries(&provider);
    assert!(matches!(
        lhs.solve(&codomain_rhs),
        Err(GenericTensorError::Facade(
            tenet::typed::Error::InvalidArgument(_)
        ))
    ));
    assert_no_provider_queries(&provider);
}

#[test]
fn checked_generic_left_solve_singular_sectors_are_nonpublishing() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let bond =
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::Vacuum, 1), (Label::X, 1)])
            .unwrap();
    let rhs: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&bond], [&bond], |_, _| 1.0).unwrap();
    for target in [Label::Vacuum, Label::X] {
        let divisor: TensorMap<_, f64> =
            TensorMap::from_block_fn(&runtime, [&bond], [&bond], |trees, _| {
                f64::from(trees.coupled() != &target)
            })
            .unwrap();
        let before = divisor.data().to_vec();
        assert!(matches!(
            divisor.solve(&rhs),
            Err(GenericTensorError::Facade(tenet::typed::Error::Operation(
                _
            )))
        ));
        assert_eq!(divisor.data(), before.as_slice());
    }
}

#[test]
fn checked_generic_left_solve_covers_all_lazy_input_pairs() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let divisor: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            if indices[0] == indices[1] {
                2.0 + indices[0] as f64
            } else {
                0.0
            }
        })
        .unwrap();
    let rhs: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            if indices[0] == indices[1] {
                2.0 + indices[0] as f64
            } else {
                1.0
            }
        })
        .unwrap();
    let expected = divisor.solve(&rhs).unwrap();
    for (lazy_lhs, lazy_rhs) in [(false, false), (true, false), (false, true), (true, true)] {
        let lhs = lazy_lhs
            .then(|| divisor.adjoint().unwrap())
            .unwrap_or_else(|| divisor.clone());
        let right = lazy_rhs
            .then(|| rhs.adjoint().unwrap())
            .unwrap_or_else(|| rhs.clone());
        reset_provider_queries(&provider);
        let solution = lhs.solve(&right).unwrap();
        assert!(std::ptr::eq(solution.provider(), provider.as_ref()));
        assert_eq!(solution.data(), expected.data());
        assert_eq!(provider.queries_since_reset.load(Ordering::Relaxed), 8);
    }
}

#[test]
fn checked_generic_solve_right_preflight_precedence_and_provider_failure_are_nonpublishing() {
    // What: error precedence and provider failures are pre-kernel, while a
    // rank-2 <- rank-1 result keeps the unequal receiver/divisor codomains.
    let solve_calls = Arc::new(AtomicUsize::new(0));
    let runtime = Runtime::builder()
        .dense_threads(1)
        .with_dense_executor(Box::new(SolveCallSpy {
            inner: DefaultDenseExecutor::default(),
            calls: Arc::clone(&solve_calls),
        }))
        .build()
        .unwrap();
    let receiver_provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let divisor_provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let receiver_leg =
        GradedSpace::try_new_with_arc(Arc::clone(&receiver_provider), [(Label::X, 2)]).unwrap();
    let divisor_leg =
        GradedSpace::try_new_with_arc(Arc::clone(&divisor_provider), [(Label::X, 2)]).unwrap();
    let receiver: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&receiver_leg], [&receiver_leg], |_, ij| {
            f64::from(ij[0] == ij[1])
        })
        .unwrap();
    let divisor: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&divisor_leg], [&divisor_leg], |_, ij| {
            if ij[0] == ij[1] {
                2.0
            } else {
                0.25
            }
        })
        .unwrap();
    let receiver_before = receiver.data().to_vec();
    let divisor_before = divisor.data().to_vec();

    let other_runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let runtime_divisor: TensorMap<_, f64> =
        TensorMap::from_block_fn(&other_runtime, [&divisor_leg], [&divisor_leg], |_, ij| {
            f64::from(ij[0] == ij[1])
        })
        .unwrap();
    reset_provider_queries(&receiver_provider);
    reset_provider_queries(&divisor_provider);
    assert!(matches!(
        receiver.solve_right(&runtime_divisor),
        Err(GenericTensorError::Facade(
            tenet::typed::Error::RuntimeMismatch
        ))
    ));
    assert_no_provider_queries(&receiver_provider);
    assert_no_provider_queries(&divisor_provider);

    let foreign_provider = Arc::new(CheckedOnlyToy::new_product_probe(1));
    let foreign_leg =
        GradedSpace::try_new_with_arc(Arc::clone(&foreign_provider), [(Label::X, 2)]).unwrap();
    let foreign_divisor: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&foreign_leg], [&foreign_leg], |_, ij| {
            f64::from(ij[0] == ij[1])
        })
        .unwrap();
    reset_provider_queries(&receiver_provider);
    reset_provider_queries(&foreign_provider);
    assert!(matches!(
        receiver.solve_right(&foreign_divisor),
        Err(GenericTensorError::Facade(
            tenet::typed::Error::RuleMismatch
        ))
    ));
    assert_no_provider_queries(&receiver_provider);
    assert_no_provider_queries(&foreign_provider);

    let wrong_domain =
        GradedSpace::try_new_with_arc(Arc::clone(&divisor_provider), [(Label::Vacuum, 1)]).unwrap();
    let domain_divisor: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&divisor_leg], [&wrong_domain], |_, _| 1.0).unwrap();
    divisor_provider.fail_algebra.store(true, Ordering::Relaxed);
    assert!(matches!(
        receiver.solve_right(&domain_divisor),
        Err(GenericTensorError::Facade(
            tenet::typed::Error::InvalidArgument(_)
        ))
    ));
    divisor_provider
        .fail_algebra
        .store(false, Ordering::Relaxed);

    let narrow =
        GradedSpace::try_new_with_arc(Arc::clone(&divisor_provider), [(Label::X, 1)]).unwrap();
    let rectangular_divisor: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&narrow], [&divisor_leg], |_, _| 1.0).unwrap();
    assert!(matches!(
        receiver.solve_right(&rectangular_divisor),
        Err(GenericTensorError::Facade(tenet::typed::Error::Operation(
            _
        )))
    ));

    divisor_provider.fail_algebra.store(true, Ordering::Relaxed);
    assert!(matches!(
        receiver.solve_right(&divisor),
        Err(GenericTensorError::Plan(
            tenet::typed::CheckedGenericPlanError::Provider(ToyError::Algebra)
        ))
    ));
    assert_eq!(solve_calls.load(Ordering::Relaxed), 0);
    divisor_provider
        .fail_algebra
        .store(false, Ordering::Relaxed);

    let wide_receiver: TensorMap<_, f64> = TensorMap::from_block_fn(
        &runtime,
        [&receiver_leg, &receiver_leg],
        [&receiver_leg],
        |_, _| 1.0,
    )
    .unwrap();
    let wide_before = wide_receiver.data().to_vec();
    receiver_provider
        .fail_algebra
        .store(true, Ordering::Relaxed);
    assert!(matches!(
        wide_receiver.solve_right(&divisor),
        Err(GenericTensorError::Plan(
            tenet::typed::CheckedGenericPlanError::Provider(ToyError::Algebra)
        ))
    ));
    assert_eq!(solve_calls.load(Ordering::Relaxed), 0);
    assert_eq!(wide_receiver.data(), wide_before.as_slice());
    assert_eq!(receiver.data(), receiver_before.as_slice());
    assert_eq!(divisor.data(), divisor_before.as_slice());
    receiver_provider
        .fail_algebra
        .store(false, Ordering::Relaxed);

    let geometry_solution = wide_receiver.solve_right(&divisor).unwrap();
    assert!(std::ptr::eq(
        geometry_solution.provider(),
        receiver_provider.as_ref()
    ));
    assert_ne!(wide_receiver.codomain(), divisor.codomain());
    assert_eq!(geometry_solution.codomain(), wide_receiver.codomain());
    assert_eq!(geometry_solution.domain(), divisor.codomain());
    assert_eq!(geometry_solution.codomain_rank(), 2);
    assert_eq!(geometry_solution.domain_rank(), 1);
    for index in 0..geometry_solution.block_count() {
        assert_eq!(geometry_solution.block(index).unwrap().shape(), [2, 2, 2]);
    }
    assert!(geometry_solution
        .compose(&divisor)
        .unwrap()
        .data()
        .iter()
        .zip(wide_receiver.data())
        .all(|(actual, expected)| (*actual - *expected).abs() < 1e-11));
}

#[test]
fn checked_generic_compact_qr_failure_is_typed_and_nonpublishing() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |_, _| 2.0).unwrap();
    let before = source.data().to_vec();
    provider.fail_algebra.store(true, Ordering::Relaxed);
    let error = source.qr_compact().unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Plan(tenet::typed::CheckedGenericPlanError::Provider(
            ToyError::Algebra
        ))
    ));
    assert_eq!(source.data(), before.as_slice());
    assert!(matches!(
        source.left_orth(),
        Err(GenericTensorError::Plan(
            tenet::typed::CheckedGenericPlanError::Provider(ToyError::Algebra)
        ))
    ));
    assert_eq!(source.data(), before.as_slice());
}

#[test]
fn checked_generic_compact_svd_failure_is_typed_and_nonpublishing() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |_, _| 2.0).unwrap();
    let before = source.data().to_vec();
    provider.fail_algebra.store(true, Ordering::Relaxed);
    let error = source.svd_compact().unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Plan(tenet::typed::CheckedGenericPlanError::Provider(
            ToyError::Algebra
        ))
    ));
    assert_eq!(source.data(), before.as_slice());
}

#[test]
fn checked_generic_compact_lq_failure_is_typed_and_nonpublishing() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |_, _| 2.0).unwrap();
    let before = source.data().to_vec();
    provider.fail_algebra.store(true, Ordering::Relaxed);
    let error = source.lq_compact().unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Plan(tenet::typed::CheckedGenericPlanError::Provider(
            ToyError::Algebra
        ))
    ));
    assert_eq!(source.data(), before.as_slice());
    assert!(matches!(
        source.right_orth(),
        Err(GenericTensorError::Plan(
            tenet::typed::CheckedGenericPlanError::Provider(ToyError::Algebra)
        ))
    ));
    assert_eq!(source.data(), before.as_slice());
}

#[test]
fn checked_generic_full_qr_failure_is_typed_and_nonpublishing() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |_, _| 2.0).unwrap();
    let before = source.data().to_vec();
    provider.fail_algebra.store(true, Ordering::Relaxed);
    let error = source.qr_full().unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Plan(tenet::typed::CheckedGenericPlanError::Provider(
            ToyError::Algebra
        ))
    ));
    assert_eq!(source.data(), before.as_slice());
}

#[test]
fn checked_generic_full_lq_reconstructs_and_preserves_provider() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |_, _| 2.0).unwrap();
    let (l, q) = source.lq_full().unwrap();
    assert!(std::ptr::eq(l.provider(), provider.as_ref()));
    assert!(std::ptr::eq(q.provider(), provider.as_ref()));
    let rebuilt = l.compose(&q).unwrap();
    assert_eq!(rebuilt.data().len(), source.data().len());
    assert!(rebuilt
        .data()
        .iter()
        .zip(source.data())
        .all(|(actual, expected)| (actual - expected).abs() < 1e-12));
}

#[test]
fn checked_generic_full_lq_supports_complex_scalars() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |_, _| {
            Complex64::new(2.0, 1.0)
        })
        .unwrap();
    let (l, q) = source.lq_full().unwrap();
    let rebuilt = l.compose(&q).unwrap();
    assert!((rebuilt.norm().unwrap() - source.norm().unwrap()).abs() < 1e-12);
}

#[test]
fn checked_generic_full_lq_failure_is_typed_and_nonpublishing() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |_, _| 2.0).unwrap();
    let before = source.data().to_vec();
    provider.fail_algebra.store(true, Ordering::Relaxed);
    let error = source.lq_full().unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Plan(tenet::typed::CheckedGenericPlanError::Provider(
            ToyError::Algebra
        ))
    ));
    assert_eq!(source.data(), before.as_slice());
}

#[test]
fn checked_only_contract_and_compose_keep_left_authority() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let left_provider = Arc::new(CheckedOnlyToy::new(0));
    let right_provider = Arc::new(CheckedOnlyToy::new(0));
    let left_leg =
        GradedSpace::try_new_with_arc(Arc::clone(&left_provider), [(Label::X, 1)]).unwrap();
    let right_leg =
        GradedSpace::try_new_with_arc(Arc::clone(&right_provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&left_leg], [&left_leg], |_, _| 1.0).unwrap();
    let nontrivial: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&left_leg, &left_leg], [&left_leg], |_, _| 1.0)
            .unwrap();
    let identity: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&right_leg], [&right_leg], |_, _| 1.0).unwrap();
    left_provider.r_queries.store(0, Ordering::Relaxed);

    for output in [
        source.contract(&identity, &[1], &[0], &[0, 1]).unwrap(),
        source.compose(&identity).unwrap(),
    ] {
        assert!(std::ptr::eq(output.provider(), left_provider.as_ref()));
        assert_eq!(output.data(), source.data());
        for index in 0..source.block_count() {
            assert_eq!(
                output.block_fusion_trees(index).unwrap(),
                source.block_fusion_trees(index).unwrap()
            );
        }
    }
    assert_eq!(left_provider.r_queries.load(Ordering::Relaxed), 0);

    let other_runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let foreign_runtime_identity: TensorMap<_, f64> =
        TensorMap::from_block_fn(&other_runtime, [&right_leg], [&right_leg], |_, _| 1.0).unwrap();
    left_provider.algebra_queries.store(0, Ordering::Relaxed);
    right_provider.algebra_queries.store(0, Ordering::Relaxed);
    assert!(matches!(
        source.contract(&foreign_runtime_identity, &[1], &[0], &[0, 1]),
        Err(GenericTensorError::Facade(
            tenet::prelude::Error::RuntimeMismatch
        ))
    ));
    assert_eq!(left_provider.algebra_queries.load(Ordering::Relaxed), 0);
    assert_eq!(right_provider.algebra_queries.load(Ordering::Relaxed), 0);

    let wrong_provider = Arc::new(CheckedOnlyToy::new(1));
    let wrong_leg =
        GradedSpace::try_new_with_arc(Arc::clone(&wrong_provider), [(Label::X, 1)]).unwrap();
    let wrong_identity: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&wrong_leg], [&wrong_leg], |_, _| 1.0).unwrap();
    left_provider.algebra_queries.store(0, Ordering::Relaxed);
    wrong_provider.algebra_queries.store(0, Ordering::Relaxed);
    assert!(source
        .contract(&wrong_identity, &[1], &[0], &[0, 1])
        .is_err());
    assert_eq!(left_provider.algebra_queries.load(Ordering::Relaxed), 0);
    assert_eq!(wrong_provider.algebra_queries.load(Ordering::Relaxed), 0);

    left_provider.fail_algebra.store(true, Ordering::Relaxed);
    let error = nontrivial
        .contract(&identity, &[2], &[0], &[0, 1, 2])
        .unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Plan(tenet::typed::CheckedGenericPlanError::Provider(
            ToyError::Algebra
        ))
    ));
}

#[test]
fn checked_only_identity_transforms_make_no_provider_queries() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();

    for counter in [
        &provider.identity_queries,
        &provider.style_queries,
        &provider.algebra_queries,
        &provider.coefficient_queries,
        &provider.f_queries,
        &provider.r_queries,
        &provider.postcommit_queries,
    ] {
        counter.store(0, Ordering::Relaxed);
    }

    for output in [
        source.permute(&[0, 1], &[2]).unwrap(),
        source.braid(&[0, 1], &[2], &[2, 1, 0]).unwrap(),
        source.transpose_axes(&[0, 1], &[2]).unwrap(),
        source.repartition(2).unwrap(),
    ] {
        assert!(std::ptr::eq(output.provider(), provider.as_ref()));
        assert_eq!(output.data().as_ptr(), source.data().as_ptr());
    }
    for counter in [
        &provider.identity_queries,
        &provider.style_queries,
        &provider.algebra_queries,
        &provider.coefficient_queries,
        &provider.f_queries,
        &provider.r_queries,
        &provider.postcommit_queries,
    ] {
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn checked_only_otimes_preserves_typed_late_f_failures() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, _| {
            trees.codomain_vertices()[0].get() as f64
                - 2.0 * trees.domain_vertices()[0].get() as f64
        })
        .unwrap();

    provider.f_queries.store(0, Ordering::Relaxed);
    source.otimes(&source).unwrap();
    let final_f_query = provider.f_queries.load(Ordering::Relaxed);
    assert!(final_f_query > 1);
    provider.f_queries.store(0, Ordering::Relaxed);
    provider.reset_commit_spy();
    provider
        .fail_f_on_query
        .store(final_f_query, Ordering::Relaxed);
    let error = source.otimes(&source).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::TensorProduct(CheckedGenericTensorProductError::Provider(
            ToyError::Algebra
        ))
    ));
    assert_eq!(provider.f_queries.load(Ordering::Relaxed), final_f_query);
    assert_eq!(provider.commit_count.load(Ordering::Relaxed), 0);
    provider.fail_f_on_query.store(0, Ordering::Relaxed);

    provider.f_queries.store(0, Ordering::Relaxed);
    provider.reset_commit_spy();
    provider.malformed_f.store(true, Ordering::Relaxed);
    let error = source.otimes(&source).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::TensorProduct(CheckedGenericTensorProductError::SymbolShape {
            symbol: "F",
            ..
        })
    ));
    assert_eq!(provider.commit_count.load(Ordering::Relaxed), 0);
    assert_eq!(provider.r_queries.load(Ordering::Relaxed), 0);
}

#[test]
fn checked_only_otimes_rejects_runtime_identity_and_style_before_algebra() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let other_runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let first = Arc::new(CheckedOnlyToy::new(0));
    let mismatched = Arc::new(CheckedOnlyToy::new(1));
    let first_leg = GradedSpace::try_new_with_arc(Arc::clone(&first), [(Label::X, 1)]).unwrap();
    let mismatched_leg =
        GradedSpace::try_new_with_arc(Arc::clone(&mismatched), [(Label::X, 1)]).unwrap();
    let lhs: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&first_leg, &first_leg], [&first_leg], |_, _| 1.0)
            .unwrap();
    let wrong_runtime: TensorMap<_, f64> = TensorMap::from_block_fn(
        &other_runtime,
        [&first_leg, &first_leg],
        [&first_leg],
        |_, _| 1.0,
    )
    .unwrap();
    let wrong_identity: TensorMap<_, f64> = TensorMap::from_block_fn(
        &runtime,
        [&mismatched_leg, &mismatched_leg],
        [&mismatched_leg],
        |_, _| 1.0,
    )
    .unwrap();
    first.algebra_queries.store(0, Ordering::Relaxed);
    first.coefficient_queries.store(0, Ordering::Relaxed);
    mismatched.algebra_queries.store(0, Ordering::Relaxed);
    mismatched.coefficient_queries.store(0, Ordering::Relaxed);

    assert!(matches!(
        lhs.otimes(&wrong_runtime),
        Err(GenericTensorError::Facade(_))
    ));
    assert!(matches!(
        lhs.otimes(&wrong_identity),
        Err(GenericTensorError::TensorProduct(
            CheckedGenericTensorProductError::Core(_)
        ))
    ));
    first.invalid_style.store(true, Ordering::Relaxed);
    assert!(matches!(
        lhs.otimes(&lhs),
        Err(GenericTensorError::TensorProduct(
            CheckedGenericTensorProductError::Core(_)
        ))
    ));
    assert_eq!(first.algebra_queries.load(Ordering::Relaxed), 0);
    assert_eq!(first.coefficient_queries.load(Ordering::Relaxed), 0);
    assert_eq!(mismatched.algebra_queries.load(Ordering::Relaxed), 0);
    assert_eq!(mismatched.coefficient_queries.load(Ordering::Relaxed), 0);
}

#[test]
fn checked_only_otimes_matches_fixed_heterogeneous_nonunit_oracle() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let first = Arc::new(CheckedOnlyToy::new_product_probe(9));
    let second = Arc::new(CheckedOnlyToy::new_product_probe(9));
    let x2 = GradedSpace::try_new_with_arc(Arc::clone(&first), [(Label::X, 2)]).unwrap();
    let x1 = GradedSpace::try_new_with_arc(Arc::clone(&first), [(Label::X, 1)]).unwrap();
    let y3 = GradedSpace::try_new_with_arc(Arc::clone(&second), [(Label::One, 3)]).unwrap();
    let y1 = GradedSpace::try_new_with_arc(Arc::clone(&second), [(Label::One, 1)]).unwrap();
    let rhs_x1 = GradedSpace::try_new_with_arc(Arc::clone(&second), [(Label::X, 1)]).unwrap();
    let lhs: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&x2, &x1], [&x1, &x1], |trees, index| {
            100.0 * trees.codomain_vertices()[0].get() as f64
                + 10.0 * trees.domain_vertices()[0].get() as f64
                + index[0] as f64
                + 1.0
        })
        .unwrap();
    let rhs: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&y3, &rhs_x1], [&y1, &rhs_x1], |_, index| {
            index[0] as f64 + 1.0
        })
        .unwrap();
    for provider in [&first, &second] {
        provider.identity_queries.store(0, Ordering::Relaxed);
        provider.style_queries.store(0, Ordering::Relaxed);
        provider.algebra_queries.store(0, Ordering::Relaxed);
        provider.coefficient_queries.store(0, Ordering::Relaxed);
        provider.r_queries.store(0, Ordering::Relaxed);
        provider.f_queries.store(0, Ordering::Relaxed);
        provider.reset_commit_spy();
    }
    let output = lhs.otimes(&rhs).unwrap();
    assert!(std::ptr::eq(output.provider(), first.as_ref()));
    assert_eq!(first.identity_queries.load(Ordering::Relaxed), 3);
    assert_eq!(second.identity_queries.load(Ordering::Relaxed), 1);
    assert_eq!(first.commit_count.load(Ordering::Relaxed), 1);
    assert_eq!(second.commit_count.load(Ordering::Relaxed), 0);
    assert_eq!(first.postcommit_queries.load(Ordering::Relaxed), 0);
    assert_eq!(second.postcommit_queries.load(Ordering::Relaxed), 0);
    assert!(first.algebra_queries.load(Ordering::Relaxed) > 0);
    assert!(first.f_queries.load(Ordering::Relaxed) > 0);
    assert_eq!(first.r_queries.load(Ordering::Relaxed), 0);
    assert_eq!(second.algebra_queries.load(Ordering::Relaxed), 0);
    assert_eq!(second.coefficient_queries.load(Ordering::Relaxed), 0);
    assert_eq!(second.f_queries.load(Ordering::Relaxed), 0);
    assert_eq!(second.r_queries.load(Ordering::Relaxed), 0);

    const EXPECTED_KEYS: [([usize; 3], [usize; 3]); 16] = [
        ([1, 1, 1], [1, 1, 1]),
        ([2, 1, 1], [1, 1, 1]),
        ([1, 1, 2], [1, 1, 1]),
        ([2, 1, 2], [1, 1, 1]),
        ([1, 1, 1], [2, 1, 1]),
        ([2, 1, 1], [2, 1, 1]),
        ([1, 1, 2], [2, 1, 1]),
        ([2, 1, 2], [2, 1, 1]),
        ([1, 1, 1], [1, 1, 2]),
        ([2, 1, 1], [1, 1, 2]),
        ([1, 1, 2], [1, 1, 2]),
        ([2, 1, 2], [1, 1, 2]),
        ([1, 1, 1], [2, 1, 2]),
        ([2, 1, 1], [2, 1, 2]),
        ([1, 1, 2], [2, 1, 2]),
        ([2, 1, 2], [2, 1, 2]),
    ];
    let keys = (0..output.block_count())
        .map(|index| {
            let trees = output.block_fusion_trees(index).unwrap();
            assert_eq!(
                trees.codomain_uncoupled(),
                &[Label::X, Label::X, Label::One, Label::X]
            );
            assert_eq!(
                trees.domain_uncoupled(),
                &[Label::X, Label::X, Label::One, Label::X]
            );
            assert_eq!(
                output.block(index).unwrap().shape(),
                &[2, 1, 3, 1, 1, 1, 1, 1]
            );
            (
                trees
                    .codomain_vertices()
                    .iter()
                    .map(|vertex| vertex.get())
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),
                trees
                    .domain_vertices()
                    .iter()
                    .map(|vertex| vertex.get())
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(keys, EXPECTED_KEYS);

    const EXPECTED_DATA: [f64; 96] = [
        555.0, 560.0, 1110.0, 1120.0, 1665.0, 1680.0, 1055.0, 1060.0, 2110.0, 2120.0, 3165.0,
        3180.0, 1221.0, 1232.0, 2442.0, 2464.0, 3663.0, 3696.0, 2321.0, 2332.0, 4642.0, 4664.0,
        6963.0, 6996.0, 605.0, 610.0, 1210.0, 1220.0, 1815.0, 1830.0, 1105.0, 1110.0, 2210.0,
        2220.0, 3315.0, 3330.0, 1331.0, 1342.0, 2662.0, 2684.0, 3993.0, 4026.0, 2431.0, 2442.0,
        4862.0, 4884.0, 7293.0, 7326.0, 1221.0, 1232.0, 2442.0, 2464.0, 3663.0, 3696.0, 2321.0,
        2332.0, 4642.0, 4664.0, 6963.0, 6996.0, 2775.0, 2800.0, 5550.0, 5600.0, 8325.0, 8400.0,
        5275.0, 5300.0, 10550.0, 10600.0, 15825.0, 15900.0, 1331.0, 1342.0, 2662.0, 2684.0, 3993.0,
        4026.0, 2431.0, 2442.0, 4862.0, 4884.0, 7293.0, 7326.0, 3025.0, 3050.0, 6050.0, 6100.0,
        9075.0, 9150.0, 5525.0, 5550.0, 11050.0, 11100.0, 16575.0, 16650.0,
    ];
    assert_eq!(output.data(), EXPECTED_DATA);

    // The first stored value has two nonzero root-multiplicity paths:
    // μ=1 contributes 111*1 and μ=2 contributes 111*4. The fixed result
    // therefore kills overwrite-instead-of-accumulate mutations.
    let colliding_path_coefficients = [1.0, 4.0];
    assert_eq!(colliding_path_coefficients.len(), 2);
    assert_eq!(
        111.0 * colliding_path_coefficients.iter().sum::<f64>(),
        EXPECTED_DATA[0]
    );

    let complex_lhs: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime, [&x2, &x1], [&x1, &x1], |trees, index| {
            Complex64::new(1.0, 1.0)
                * (100.0 * trees.codomain_vertices()[0].get() as f64
                    + 10.0 * trees.domain_vertices()[0].get() as f64
                    + index[0] as f64
                    + 1.0)
        })
        .unwrap();
    let complex_rhs: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime, [&y3, &rhs_x1], [&y1, &rhs_x1], |_, index| {
            Complex64::new(2.0, -3.0) * (index[0] as f64 + 1.0)
        })
        .unwrap();
    first.f_queries.store(0, Ordering::Relaxed);
    first.reset_commit_spy();
    let complex = complex_lhs.otimes(&complex_rhs).unwrap();
    assert!(std::ptr::eq(complex.provider(), first.as_ref()));
    for (actual, expected) in complex.data().iter().zip(EXPECTED_DATA) {
        assert!((*actual - Complex64::new(5.0, -1.0) * expected).norm() <= 1e-12);
    }
}

#[test]
fn checked_errors_stay_typed_and_callback_waits_for_all_decodes() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let error =
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::Invalid, 1)]).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Structure(CheckedGenericStructureError::Provider(
            ToyError::InvalidSector
        ))
    ));

    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    provider.fail_decode.store(true, Ordering::Relaxed);
    let callbacks = AtomicUsize::new(0);
    let error = TensorMap::<_, f64>::from_block_fn(&runtime, [&leg, &leg], [&leg], |_, _| {
        callbacks.fetch_add(1, Ordering::Relaxed);
        1.0
    })
    .unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Structure(CheckedGenericStructureError::Provider(ToyError::Decode))
    ));
    assert_eq!(callbacks.load(Ordering::Relaxed), 0);
}

#[test]
fn identity_mismatch_precedes_algebra_queries_and_both_dtypes_fill() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let first = Arc::new(CheckedOnlyToy::new(0));
    let other = Arc::new(CheckedOnlyToy::new(1));
    let left = GradedSpace::try_new_with_arc(Arc::clone(&first), [(Label::X, 1)]).unwrap();
    let right = GradedSpace::try_new_with_arc(Arc::clone(&other), [(Label::X, 1)]).unwrap();
    let error = TensorMap::<_, f64>::zeros(&runtime, [&left], [&right]).unwrap_err();
    assert!(matches!(error, GenericTensorError::Facade(_)));
    assert_eq!(first.algebra_queries.load(Ordering::Relaxed), 0);
    assert_eq!(other.algebra_queries.load(Ordering::Relaxed), 0);

    let real: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&runtime, [&left, &left], [&left], 7).unwrap();
    let complex: TensorMap<_, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&left, &left], [&left], 7).unwrap();
    assert_eq!(real.data().len(), complex.data().len());
    assert!(complex.data().iter().any(|value| value.im != 0.0));
}

#[test]
fn failed_checked_admission_does_not_advance_the_runtime_stream() {
    let runtime_a = Runtime::builder().dense_threads(1).build().unwrap();
    let runtime_b = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(7));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 1)]).unwrap();

    provider.fail_algebra.store(true, Ordering::Relaxed);
    assert!(TensorMap::<_, f64>::rand(&runtime_a, [&leg, &leg], [&leg]).is_err());
    provider.fail_algebra.store(false, Ordering::Relaxed);

    let after_failure = TensorMap::<_, f64>::rand(&runtime_a, [&leg, &leg], [&leg]).unwrap();
    let control = TensorMap::<_, f64>::rand(&runtime_b, [&leg, &leg], [&leg]).unwrap();
    assert_eq!(after_failure.data(), control.data());
}

#[test]
fn checked_generic_cat_admits_once_and_queries_only_left_before_commit() {
    // What: successful catdomain admission uses the left provider Arc once;
    // admitted identity stamps keep the equal-identity right provider cold,
    // and copy planning performs no provider query after commit.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let left_provider = Arc::new(CheckedOnlyToy::new(0));
    let right_provider = Arc::new(CheckedOnlyToy::new(0));
    let left_common =
        GradedSpace::try_new_with_arc(Arc::clone(&left_provider), [(Label::X, 1)]).unwrap();
    let right_common =
        GradedSpace::try_new_with_arc(Arc::clone(&right_provider), [(Label::X, 1)]).unwrap();
    let left_changed =
        GradedSpace::try_new_with_arc(Arc::clone(&left_provider), [(Label::X, 1)]).unwrap();
    let right_changed =
        GradedSpace::try_new_with_arc(Arc::clone(&right_provider), [(Label::X, 2)]).unwrap();
    let lhs: TensorMap<_, f64> = TensorMap::from_block_fn(
        &runtime,
        [&left_common, &left_common],
        [&left_changed],
        |trees, indices| {
            10.0 + trees.codomain_vertices()[0].get() as f64 + indices.iter().sum::<usize>() as f64
        },
    )
    .unwrap();
    let rhs: TensorMap<_, f64> = TensorMap::from_block_fn(
        &runtime,
        [&right_common, &right_common],
        [&right_changed],
        |trees, indices| {
            20.0 + trees.codomain_vertices()[0].get() as f64 + indices.iter().sum::<usize>() as f64
        },
    )
    .unwrap();
    let combined = left_changed.oplus(&right_changed).unwrap();
    for provider in [&left_provider, &right_provider] {
        for counter in [
            &provider.identity_queries,
            &provider.style_queries,
            &provider.algebra_queries,
            &provider.coefficient_queries,
            &provider.f_queries,
            &provider.r_queries,
        ] {
            counter.store(0, Ordering::Relaxed);
        }
        provider.reset_commit_spy();
    }
    let _: TensorMap<_, f64> =
        TensorMap::zeros(&runtime, [&left_common, &left_common], [&combined]).unwrap();
    let mut admission_queries = [
        left_provider.identity_queries.load(Ordering::Relaxed),
        left_provider.style_queries.load(Ordering::Relaxed),
        left_provider.algebra_queries.load(Ordering::Relaxed),
        left_provider.coefficient_queries.load(Ordering::Relaxed),
        left_provider.f_queries.load(Ordering::Relaxed),
        left_provider.r_queries.load(Ordering::Relaxed),
    ];
    let admission_query_count = left_provider.queries_since_reset.load(Ordering::Relaxed) - 3;
    // `zeros` first checks the three supplied leg authorities; cat starts from
    // already-admitted stamps, so remove exactly those three identity reads.
    admission_queries[0] -= 3;
    for counter in [
        &left_provider.identity_queries,
        &left_provider.style_queries,
        &left_provider.algebra_queries,
        &left_provider.coefficient_queries,
        &left_provider.f_queries,
        &left_provider.r_queries,
    ] {
        counter.store(0, Ordering::Relaxed);
    }
    left_provider.arm_commit_spy_after_queries(admission_query_count);

    let output = lhs.catdomain(&rhs).unwrap();

    assert!(std::ptr::eq(output.provider(), left_provider.as_ref()));
    assert!(!std::ptr::eq(output.provider(), right_provider.as_ref()));
    assert_eq!(left_provider.commit_count.load(Ordering::Relaxed), 1);
    assert_eq!(left_provider.postcommit_queries.load(Ordering::Relaxed), 0);
    assert_eq!(
        [
            left_provider.identity_queries.load(Ordering::Relaxed),
            left_provider.style_queries.load(Ordering::Relaxed),
            left_provider.algebra_queries.load(Ordering::Relaxed),
            left_provider.coefficient_queries.load(Ordering::Relaxed),
            left_provider.f_queries.load(Ordering::Relaxed),
            left_provider.r_queries.load(Ordering::Relaxed),
        ],
        admission_queries
    );
    for counter in [
        &right_provider.identity_queries,
        &right_provider.style_queries,
        &right_provider.algebra_queries,
        &right_provider.coefficient_queries,
        &right_provider.f_queries,
        &right_provider.r_queries,
        &right_provider.postcommit_queries,
    ] {
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn checked_generic_cat_precedence_and_admission_failure_are_typed_nonpublishing() {
    // What: admission stamps, runtime, cat arguments, then output admission
    // reject in order; every failure leaves both admitted input payloads alone.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let other_runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let equal = Arc::new(CheckedOnlyToy::new(0));
    let wrong = Arc::new(CheckedOnlyToy::new(1));
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let equal_leg = GradedSpace::try_new_with_arc(Arc::clone(&equal), [(Label::X, 1)]).unwrap();
    let wrong_leg = GradedSpace::try_new_with_arc(Arc::clone(&wrong), [(Label::X, 1)]).unwrap();
    let lhs: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |_, indices| {
            1.0 + indices.iter().sum::<usize>() as f64
        })
        .unwrap();
    let valid_rhs: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&equal_leg, &equal_leg], [&equal_leg], |_, _| 2.0)
            .unwrap();
    let wrong_identity: TensorMap<_, f64> = TensorMap::zeros(
        &other_runtime,
        [&wrong_leg, &wrong_leg],
        [&wrong_leg, &wrong_leg],
    )
    .unwrap();
    let wrong_runtime: TensorMap<_, f64> = TensorMap::zeros(
        &other_runtime,
        [&equal_leg, &equal_leg],
        [&equal_leg, &equal_leg],
    )
    .unwrap();
    let bad_arguments: TensorMap<_, f64> =
        TensorMap::zeros(&runtime, [&equal_leg, &equal_leg], [&equal_leg, &equal_leg]).unwrap();
    let lhs_before = lhs.data().to_vec();
    let rhs_before = valid_rhs.data().to_vec();
    provider.fail_algebra.store(true, Ordering::Relaxed);
    for counter in [
        &provider.identity_queries,
        &provider.style_queries,
        &provider.algebra_queries,
        &provider.coefficient_queries,
        &provider.f_queries,
        &provider.r_queries,
    ] {
        counter.store(0, Ordering::Relaxed);
    }
    provider.reset_commit_spy();

    assert!(matches!(
        lhs.catdomain(&wrong_identity),
        Err(GenericTensorError::Facade(
            tenet::prelude::Error::RuleMismatch
        ))
    ));
    assert!(matches!(
        lhs.catdomain(&wrong_runtime),
        Err(GenericTensorError::Facade(
            tenet::prelude::Error::RuntimeMismatch
        ))
    ));
    assert!(matches!(
        lhs.catdomain(&bad_arguments),
        Err(GenericTensorError::Facade(
            tenet::prelude::Error::InvalidArgument(_)
        ))
    ));
    assert_eq!(provider.identity_queries.load(Ordering::Relaxed), 0);
    assert_eq!(provider.algebra_queries.load(Ordering::Relaxed), 0);

    assert!(matches!(
        lhs.catdomain(&valid_rhs),
        Err(GenericTensorError::Structure(
            CheckedGenericStructureError::Provider(ToyError::Algebra)
        ))
    ));
    assert_eq!(provider.commit_count.load(Ordering::Relaxed), 0);
    assert_eq!(lhs.data(), lhs_before);
    assert_eq!(valid_rhs.data(), rhs_before);
}

#[cfg(feature = "racah-generated")]
fn sun_cat_marker(trees: &tenet::typed::BlockFusionTrees<Vec<i64>>) -> usize {
    trees
        .codomain_vertices()
        .iter()
        .enumerate()
        .map(|(index, vertex)| (index + 1) * 100 * vertex.get())
        .chain(
            trees
                .domain_vertices()
                .iter()
                .enumerate()
                .map(|(index, vertex)| (index + 1) * 1_000 * vertex.get()),
        )
        .sum()
}

#[cfg(feature = "racah-generated")]
fn assert_sun_cat_values<D>(
    output: &TensorMap<tenet::typed::SUNFusionRule, D>,
    lhs: &TensorMap<tenet::typed::SUNFusionRule, D>,
    rhs: &TensorMap<tenet::typed::SUNFusionRule, D>,
    changed_axis: usize,
    lhs_extent: usize,
    value: impl Fn(usize) -> D,
) where
    D: Copy + fmt::Debug + PartialEq + tenet::typed::TensorScalar,
{
    let mut saw_mu_two = false;
    for output_index in 0..output.block_count() {
        let trees = output.block_fusion_trees(output_index).unwrap();
        saw_mu_two |= trees
            .codomain_vertices()
            .iter()
            .chain(trees.domain_vertices())
            .any(|vertex| vertex.get() == 2);
        assert!((0..lhs.block_count()).any(|index| lhs.block_fusion_trees(index).unwrap() == trees));
        assert!((0..rhs.block_count()).any(|index| rhs.block_fusion_trees(index).unwrap() == trees));
        let block = output.block(output_index).unwrap();
        let elements = block.shape().iter().product::<usize>();
        for linear in 0..elements {
            let mut remainder = linear;
            let mut indices = Vec::with_capacity(block.shape().len());
            let mut position = block.offset();
            for (&extent, &stride) in block.shape().iter().zip(block.strides()) {
                let index = remainder % extent;
                remainder /= extent;
                indices.push(index);
                position += index * stride;
            }
            let (base, local_changed) = if indices[changed_axis] < lhs_extent {
                (10_000, indices[changed_axis])
            } else {
                (20_000, indices[changed_axis] - lhs_extent)
            };
            indices[changed_axis] = local_changed;
            let index_marker = indices
                .iter()
                .enumerate()
                .map(|(axis, index)| (axis + 1) * index)
                .sum::<usize>();
            assert_eq!(
                output.data()[position],
                value(base + sun_cat_marker(&trees) + index_marker)
            );
        }
    }
    assert!(saw_mu_two, "SU(N) cat fixture must carry a μ=2 full key");
}

#[cfg(feature = "racah-generated")]
fn assert_sun_cat_case<D>(n: usize, label: Vec<i64>, value: impl Fn(usize) -> D + Copy)
where
    D: Copy + fmt::Debug + PartialEq + tenet::typed::TensorScalar,
{
    use tenet::typed::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let left_provider = Arc::new(SUNFusionRule::new(n).unwrap());
    let right_provider = Arc::new(SUNFusionRule::new(n).unwrap());
    let left_common =
        GradedSpace::try_new_with_arc(Arc::clone(&left_provider), [(label.clone(), 1)]).unwrap();
    let right_common =
        GradedSpace::try_new_with_arc(Arc::clone(&right_provider), [(label.clone(), 1)]).unwrap();
    let left_changed =
        GradedSpace::try_new_with_arc(Arc::clone(&left_provider), [(label.clone(), 1)]).unwrap();
    let right_changed =
        GradedSpace::try_new_with_arc(Arc::clone(&right_provider), [(label.clone(), 2)]).unwrap();
    let fill = |base, trees: &tenet::typed::BlockFusionTrees<Vec<i64>>, indices: &[usize]| {
        value(
            base + sun_cat_marker(trees)
                + indices
                    .iter()
                    .enumerate()
                    .map(|(axis, index)| (axis + 1) * index)
                    .sum::<usize>(),
        )
    };

    let domain_lhs: TensorMap<_, D> = TensorMap::from_block_fn(
        &runtime,
        [&left_common, &left_common],
        [&left_changed],
        |trees, indices| fill(10_000, trees, indices),
    )
    .unwrap();
    let domain_rhs: TensorMap<_, D> = TensorMap::from_block_fn(
        &runtime,
        [&right_common, &right_common],
        [&right_changed],
        |trees, indices| fill(20_000, trees, indices),
    )
    .unwrap();
    let domain = domain_lhs.catdomain(&domain_rhs).unwrap();
    assert!(std::ptr::eq(domain.provider(), left_provider.as_ref()));
    assert!(!std::ptr::eq(domain.provider(), right_provider.as_ref()));
    assert_eq!(domain.domain()[0].degeneracy(&label).unwrap(), 3);
    assert_sun_cat_values(&domain, &domain_lhs, &domain_rhs, 2, 1, value);
    let lazy_domain = domain_lhs
        .adjoint()
        .unwrap()
        .catcodomain(&domain_rhs.adjoint().unwrap())
        .unwrap();
    assert_eq!(lazy_domain.data(), domain.adjoint().unwrap().data());

    let codomain_lhs: TensorMap<_, D> = TensorMap::from_block_fn(
        &runtime,
        [&left_changed],
        [&left_common, &left_common],
        |trees, indices| fill(10_000, trees, indices),
    )
    .unwrap();
    let codomain_rhs: TensorMap<_, D> = TensorMap::from_block_fn(
        &runtime,
        [&right_changed],
        [&right_common, &right_common],
        |trees, indices| fill(20_000, trees, indices),
    )
    .unwrap();
    let codomain = codomain_lhs.catcodomain(&codomain_rhs).unwrap();
    assert!(std::ptr::eq(codomain.provider(), left_provider.as_ref()));
    assert_eq!(codomain.codomain()[0].degeneracy(&label).unwrap(), 3);
    assert_sun_cat_values(&codomain, &codomain_lhs, &codomain_rhs, 0, 1, value);
    let lazy_codomain = codomain_lhs
        .adjoint()
        .unwrap()
        .catdomain(&codomain_rhs.adjoint().unwrap())
        .unwrap();
    assert_eq!(lazy_codomain.data(), codomain.adjoint().unwrap().data());
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_checked_generic_cat_covers_both_directions_dtypes_and_mu_two_keys() {
    // What: exact TensorKit direct-sum slab values, μ=2 full-key matching,
    // distinct equal-identity Arcs, left authority, and lazy-adjoint parity.
    for (n, label) in [(3, vec![1, 1]), (4, vec![1, 0, 1])] {
        assert_sun_cat_case(n, label.clone(), |value| value as f64);
        assert_sun_cat_case(n, label, |value| {
            Complex64::new(value as f64, -(value as f64))
        });
    }
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_adjoint_multiplicity_transforms_round_trip_labels_vertices_and_payload() {
    use tenet::typed::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    for (n, adjoint) in [(3, vec![1, 1]), (4, vec![1, 0, 1])] {
        let provider = Arc::new(SUNFusionRule::new(n).unwrap());
        let leg =
            GradedSpace::try_new_with_arc(Arc::clone(&provider), [(adjoint.clone(), 1)]).unwrap();
        let tensor: TensorMap<_, f64> =
            TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |trees, _| {
                trees.codomain_vertices()[0].get() as f64
            })
            .unwrap();
        assert_eq!(tensor.block_count(), 2);
        for index in 0..2 {
            let trees = tensor.block_fusion_trees(index).unwrap();
            assert_eq!(trees.coupled(), &adjoint);
            assert_eq!(
                trees.codomain_uncoupled(),
                &[adjoint.clone(), adjoint.clone()]
            );
            assert_eq!(trees.domain_uncoupled(), std::slice::from_ref(&adjoint));
            assert_eq!(trees.codomain_vertices()[0].get(), index + 1);
            assert_eq!(tensor.block(index).unwrap().shape(), &[1, 1, 1]);
        }
        assert_eq!(tensor.data(), &[1.0, 2.0]);

        let identity: TensorMap<_, f64> =
            TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, _| 1.0).unwrap();
        for output in [
            tensor.contract(&identity, &[2], &[0], &[0, 1, 2]).unwrap(),
            tensor.compose(&identity).unwrap(),
        ] {
            assert!(std::ptr::eq(output.provider(), provider.as_ref()));
            assert_eq!(output.data(), tensor.data());
            for index in 0..tensor.block_count() {
                assert_eq!(
                    output.block_fusion_trees(index).unwrap(),
                    tensor.block_fusion_trees(index).unwrap()
                );
            }
        }

        let product = tensor.otimes(&tensor).unwrap();
        assert!(std::ptr::eq(product.provider(), provider.as_ref()));
        let (expected_len, expected_sum, expected_weighted, expected_prefix): (
            usize,
            f64,
            f64,
            &[f64],
        ) = match n {
            3 => (
                145,
                9.468_841_418_575_323,
                39.231_504_693_264_13,
                &[
                    0.0,
                    1.0,
                    2.0,
                    2.0,
                    4.0,
                    0.0,
                    0.0,
                    0.0,
                    0.353_553_390_593_273_6,
                    0.707_106_781_186_547_2,
                    0.0,
                    0.857_142_857_142_857,
                ],
            ),
            4 => (
                245,
                8.608_165_620_335_726,
                -1.317_392_645_553_582,
                &[
                    0.0,
                    0.0,
                    1.0,
                    2.0,
                    2.0,
                    4.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.338_873_675_850_995_87,
                    0.677_747_351_701_991_7,
                ],
            ),
            _ => unreachable!(),
        };
        assert_eq!(product.data().len(), expected_len);
        let sum = product.data().iter().sum::<f64>();
        let weighted = product
            .data()
            .iter()
            .enumerate()
            .map(|(index, value)| (index + 1) as f64 * value)
            .sum::<f64>();
        assert!((sum - expected_sum).abs() <= 1e-10);
        assert!((weighted - expected_weighted).abs() <= 1e-9);
        for (&actual, &expected) in product.data().iter().zip(expected_prefix) {
            assert!((actual - expected).abs() <= 1e-10);
        }
        let mut adjoint_root_vertices = Vec::new();
        for index in 0..product.block_count() {
            let trees = product.block_fusion_trees(index).unwrap();
            if trees.coupled() == &adjoint
                && product.data()[product.block(index).unwrap().offset()].abs() > 1e-10
            {
                assert_eq!(trees.codomain_uncoupled(), vec![adjoint.clone(); 4]);
                assert_eq!(trees.domain_uncoupled(), vec![adjoint.clone(); 2]);
                adjoint_root_vertices.push(trees.domain_vertices().last().unwrap().get());
            }
        }
        adjoint_root_vertices.sort_unstable();
        adjoint_root_vertices.dedup();
        assert_eq!(adjoint_root_vertices, [1, 2]);

        let snapshot = |tensor: &TensorMap<SUNFusionRule, f64>| {
            (0..tensor.block_count())
                .map(|index| tensor.block_fusion_trees(index).unwrap())
                .collect::<Vec<_>>()
        };
        let source_snapshot = snapshot(&tensor);
        for restored in [
            tensor
                .permute(&[1, 0], &[2])
                .unwrap()
                .permute(&[1, 0], &[2])
                .unwrap(),
            tensor
                .braid(&[0, 2], &[1], &[0, 1, 2])
                .unwrap()
                .braid(&[0, 2], &[1], &[0, 1, 2])
                .unwrap(),
            tensor.repartition(1).unwrap().repartition(2).unwrap(),
            tensor.transpose().unwrap().transpose().unwrap(),
        ] {
            assert!(std::ptr::eq(restored.provider(), provider.as_ref()));
            assert_eq!(snapshot(&restored), source_snapshot);
            for (actual, expected) in restored.data().iter().zip(tensor.data()) {
                assert!((actual - expected).abs() <= 1e-10);
            }
        }
    }
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_checked_generic_adjoint_and_reductions_preserve_provider_and_errors() {
    use tenet::core::SUNFusionRuleError;
    use tenet::typed::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    for (n, adjoint) in [(3, vec![1, 1]), (4, vec![1, 0, 1])] {
        let provider = Arc::new(SUNFusionRule::new(n).unwrap());
        let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(adjoint, 1)]).unwrap();
        let source: TensorMap<_, f64> =
            TensorMap::from_block_fn(&runtime, [&leg], [&leg], |trees, _| {
                trees.coupled().iter().sum::<i64>() as f64 + 1.0
            })
            .unwrap();

        let dagger = source.adjoint().unwrap();
        assert!(std::ptr::eq(dagger.provider(), provider.as_ref()));
        assert!((dagger.norm().unwrap() - source.norm().unwrap()).abs() < 1.0e-12);
        assert!((dagger.inner(&dagger).unwrap() - source.inner(&source).unwrap()).abs() < 1.0e-12);
        assert!((dagger.tr().unwrap() - source.tr().unwrap()).abs() < 1.0e-12);

        let complex = source.to_c64();
        let complex_dagger = complex.adjoint().unwrap();
        assert!(std::ptr::eq(complex_dagger.provider(), provider.as_ref()));
        assert!((complex_dagger.tr().unwrap() - complex.tr().unwrap().conj()).norm() < 1.0e-12);
    }

    let provider = SUNFusionRule::new(3).unwrap();
    let three = provider.encode_dynkin(&[1, 0]).unwrap();
    let eight = provider.encode_dynkin(&[1, 1]).unwrap();
    assert!(matches!(
        provider.try_r_symbol_generic(three, three, eight),
        Err(SUNFusionRuleError::Racah(_))
    ));
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_checked_generic_transforms_reuse_the_runtime_completed_store() {
    use tenet::typed::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    for (n, adjoint) in [(3, vec![1, 1]), (4, vec![1, 0, 1])] {
        let provider = Arc::new(SUNFusionRule::new(n).unwrap());
        let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(adjoint, 1)]).unwrap();
        let source: TensorMap<_, f64> =
            TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |trees, _| {
                trees.codomain_vertices()[0].get() as f64
            })
            .unwrap();

        for operation in ["permute", "braid", "repartition"] {
            runtime.clear_tree_transform_cache();
            let apply = |tensor: &TensorMap<SUNFusionRule, f64>| match operation {
                "permute" => tensor.permute(&[1, 0], &[2]),
                "braid" => tensor.braid(&[1, 0], &[2], &[0, 1, 2]),
                "repartition" => tensor.repartition(1),
                _ => unreachable!(),
            };
            let first = apply(&source).unwrap();
            let cold = runtime.tree_transform_cache_info();
            let repeated = apply(&source).unwrap();
            let warm = runtime.tree_transform_cache_info();

            assert_eq!(cold.entries(), 1);
            assert_eq!(cold.misses(), 1);
            assert_eq!(warm.entries(), 1);
            assert_eq!(warm.misses(), 1);
            assert_eq!(warm.hits(), 1);
            assert_eq!(repeated.data(), first.data());
            assert!(std::ptr::eq(first.provider(), provider.as_ref()));
            assert!(std::ptr::eq(repeated.provider(), provider.as_ref()));
            for index in 0..first.block_count() {
                assert_eq!(
                    repeated.block_fusion_trees(index).unwrap(),
                    first.block_fusion_trees(index).unwrap()
                );
            }
        }
    }
}
