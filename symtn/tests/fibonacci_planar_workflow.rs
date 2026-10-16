//! The public planar Fibonacci workflow (#633) on both Fibonacci providers,
//! against a TensorKit golden-chain oracle. Moved from the deleted
//! `symtn-category-data` crate (#2016).

use std::sync::Arc;

use num_complex::Complex64;
use symtn::typed::{ContractSpec, GradedSpace, Runtime, TensorMap};
use symtn_sectors::{
    CheckedFusionAlgebra, FibonacciFusionRule, FibonacciSector, FibonacciTableRule,
    MultiplicityFreeRigidSymbols, SectorCodec,
};

/// One fixed bound for the planar workflow. At most three unit-norm 2x2
/// transforms and two 2x2 products contribute to any checked coordinate, so
/// `256 * f64::EPSILON` leaves margin for the imported decimals and BLAS
/// arithmetic without making the bound provider-tunable.
const PLANAR_BOUND: f64 = 256.0 * f64::EPSILON;

type Matrix2 = [[Complex64; 2]; 2];

struct PlanarOracle {
    f: Matrix2,
    sigma1: Matrix2,
    sigma2: Matrix2,
    yang_baxter: Matrix2,
    hamiltonian: Matrix2,
}

fn matrix_product(lhs: Matrix2, rhs: Matrix2) -> Matrix2 {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            (0..2)
                .map(|inner| lhs[row][inner] * rhs[inner][column])
                .sum()
        })
    })
}

fn matrix_adjoint(matrix: Matrix2) -> Matrix2 {
    std::array::from_fn(|row| std::array::from_fn(|column| matrix[column][row].conj()))
}

fn assert_matrix_close(actual: Matrix2, expected: Matrix2, what: &str) {
    for row in 0..2 {
        for column in 0..2 {
            let residual = (actual[row][column] - expected[row][column]).norm();
            assert!(
                residual <= PLANAR_BOUND,
                "{what}[{row},{column}] residual {residual:e}: {:?} != {:?}",
                actual[row][column],
                expected[row][column]
            );
        }
    }
}

fn coordinates<R>(
    state: &TensorMap<R, Complex64>,
    vacuum: R::Sector,
    tau: R::Sector,
) -> [Complex64; 2]
where
    R: MultiplicityFreeRigidSymbols<Scalar = Complex64> + CheckedFusionAlgebra + SectorCodec,
    R::Sector: Copy,
{
    let mut values = [Complex64::new(0.0, 0.0); 2];
    for (trees, block) in state.subblocks().unwrap() {
        let slot = match trees.codomain_innerlines() {
            [channel] if *channel == vacuum => 0,
            [channel] if *channel == tau => 1,
            inner => panic!("unexpected Fibonacci fusion path {inner:?}"),
        };
        values[slot] = *block.get(&[0, 0, 0, 0]).unwrap();
    }
    values
}

fn sigma<R>(
    state: &TensorMap<R, Complex64>,
    generator: usize,
    inverse: bool,
) -> TensorMap<R, Complex64>
where
    R: MultiplicityFreeRigidSymbols<Scalar = Complex64> + CheckedFusionAlgebra + SectorCodec,
{
    let codomain = if generator == 1 { [1, 0, 2] } else { [0, 2, 1] };
    // Source axes are [tau_1, tau_2, tau_3 | total tau]. Increasing levels
    // define positive Artin generators; reversing them selects the inverse.
    // Why not `permute`: an anyonic swap without source levels loses the
    // over/under convention and is not this braid word.
    let levels = if inverse { [3, 2, 1, 0] } else { [0, 1, 2, 3] };
    state.braid(&codomain, &[3], &levels).unwrap()
}

fn planar_fibonacci_workflow<R>(
    provider: R,
    vacuum: R::Sector,
    tau_sector: R::Sector,
) -> PlanarOracle
where
    R: MultiplicityFreeRigidSymbols<Scalar = Complex64> + CheckedFusionAlgebra + SectorCodec,
    R::Sector: Copy,
{
    let runtime = Runtime::builder().build().unwrap();
    let tau = GradedSpace::try_new(Arc::new(provider), [(tau_sector, 1)]).unwrap();
    let dual_tau = tau.try_dual().unwrap();

    for (rank, expected) in [(2, 1), (3, 2), (4, 3)] {
        let legs = vec![&tau; rank];
        let space: TensorMap<_, Complex64> = TensorMap::zeros(&runtime, legs, [&tau]).unwrap();
        assert_eq!(
            space.dense_data().unwrap().len(),
            expected,
            "all-tau rank {rank}"
        );
    }

    let basis: [TensorMap<_, Complex64>; 2] = [vacuum, tau_sector].map(|channel| {
        TensorMap::from_subblock_fn(&runtime, [&tau, &tau, &tau], [&tau], |trees, _| {
            Complex64::new((trees.codomain_innerlines() == [channel]) as u8 as f64, 0.0)
        })
        .unwrap()
    });
    let ordinary = basis[0]
        .contract(
            &basis[0],
            &ContractSpec {
                lhs: &[3],
                rhs: &[0],
                codomain: &[0, 1, 2],
                domain: &[3, 4, 5],
            },
        )
        .unwrap_err();
    assert!(matches!(
        ordinary,
        symtn::typed::Error::Operation(operation)
            if matches!(*operation, symtn::typed::OperationError::UnsupportedTensorContractScope { .. })
    ));
    let dual_basis: [TensorMap<_, Complex64>; 2] = [vacuum, tau_sector].map(|channel| {
        TensorMap::from_subblock_fn(
            &runtime,
            [&dual_tau, &dual_tau, &dual_tau],
            [&dual_tau],
            |trees, _| Complex64::new((trees.codomain_innerlines() == [channel]) as u8 as f64, 0.0),
        )
        .unwrap()
    });

    // `transpose` is the crossing-free boundary move. Its two basis columns
    // reconstruct F; applying the same planar move again is its inverse.
    let f = std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            let transformed = basis[column].transpose(&[3], &[2, 1, 0]).unwrap();
            assert_eq!(transformed.dense_data().unwrap().len(), 2);
            let roundtrip = transformed.transpose(&[3, 2, 1], &[0]).unwrap();
            for (slot, value) in coordinates(&roundtrip, vacuum, tau_sector)
                .into_iter()
                .enumerate()
            {
                let expected = Complex64::new((slot == column) as u8 as f64, 0.0);
                assert!((value - expected).norm() <= PLANAR_BOUND);
            }
            // Exact-boundary composition only: (tau <- tau^3) o
            // (tau^3 <- tau). Why not cross providers: their RuleIdentity
            // values intentionally differ, so each workflow stays bound to
            // its own provider allocation.
            transformed
                .compose(&dual_basis[row])
                .unwrap()
                .dense_data()
                .unwrap()[0]
        })
    });

    let generator_matrix = |generator| {
        std::array::from_fn(|row| {
            std::array::from_fn(|column| {
                coordinates(&sigma(&basis[column], generator, false), vacuum, tau_sector)[row]
            })
        })
    };
    let inverse_matrix = |generator| {
        std::array::from_fn(|row| {
            std::array::from_fn(|column| {
                coordinates(&sigma(&basis[column], generator, true), vacuum, tau_sector)[row]
            })
        })
    };
    let sigma1 = generator_matrix(1);
    let sigma2 = generator_matrix(2);
    let identity = [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
    ];
    for (generator, matrix) in [(1, sigma1), (2, sigma2)] {
        let inverse = inverse_matrix(generator);
        assert_matrix_close(inverse, matrix_adjoint(matrix), "braid inverse");
        assert_matrix_close(
            matrix_product(inverse, matrix),
            identity,
            "generator unitarity",
        );
        assert_matrix_close(
            matrix_product(matrix, inverse),
            identity,
            "generator inverse",
        );
    }

    let braid_word = |word: [usize; 3]| {
        std::array::from_fn(|row| {
            std::array::from_fn(|column| {
                let state = word
                    .into_iter()
                    .fold(basis[column].clone(), |state, generator| {
                        sigma(&state, generator, false)
                    });
                coordinates(&state, vacuum, tau_sector)[row]
            })
        })
    };
    let yang_baxter = braid_word([1, 2, 1]);
    assert_matrix_close(yang_baxter, braid_word([2, 1, 2]), "Yang--Baxter");

    let projector = [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
    ];
    let second_projector = matrix_product(matrix_product(f, projector), matrix_adjoint(f));
    let hamiltonian = std::array::from_fn(|row| {
        std::array::from_fn(|column| -(projector[row][column] + second_projector[row][column]))
    });

    PlanarOracle {
        f,
        sigma1,
        sigma2,
        yang_baxter,
        hamiltonian,
    }
}

#[test]
fn fibonacci_public_planar_workflow_matches_both_providers_and_tensorkit() {
    let closed = planar_fibonacci_workflow(
        FibonacciFusionRule,
        FibonacciSector::Vacuum,
        FibonacciSector::Tau,
    );
    let imported = planar_fibonacci_workflow(
        FibonacciTableRule::try_new().expect("the shipped tables load"),
        FibonacciSector::Vacuum,
        FibonacciSector::Tau,
    );

    assert_matrix_close(imported.f, closed.f, "provider F");
    assert_matrix_close(imported.sigma1, closed.sigma1, "provider sigma1");
    assert_matrix_close(imported.sigma2, closed.sigma2, "provider sigma2");
    assert_matrix_close(
        imported.yang_baxter,
        closed.yang_baxter,
        "provider final state",
    );

    // TensorKit 0.17 at cfaa073e4d1e3eb2167edcbdc3be9872f41e7d91 with
    // TensorKitSectors + CategoryData, pinned in the
    // vacuum-first left-associated basis [(tau tau)->1, (tau tau)->tau],
    // source order [tau_1,tau_2,tau_3 | total tau], and the gauge recorded in
    // `fixtures/fib-categorydata-v0.1.3.txt`. The braid oracle uses the
    // positive words sigma_1 sigma_2 sigma_1 and sigma_2 sigma_1 sigma_2 with
    // increasing source levels [0,1,2,3] (reversed for the inverse). This is
    // H = -(P_12 + F P_12 F^dagger), not a non-planar ordinary contraction.
    let tensorkit_hamiltonian = [
        [
            Complex64::new(-1.381966011250105, 0.0),
            Complex64::new(-0.48586827175664565, 0.0),
        ],
        [
            Complex64::new(-0.48586827175664565, 0.0),
            Complex64::new(-0.6180339887498948, 0.0),
        ],
    ];
    assert_matrix_close(
        closed.hamiltonian,
        tensorkit_hamiltonian,
        "closed golden chain",
    );
    assert_matrix_close(
        imported.hamiltonian,
        tensorkit_hamiltonian,
        "table golden chain",
    );
}
