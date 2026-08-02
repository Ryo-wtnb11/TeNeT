use std::hint::black_box;
use std::time::Instant;

use tenet::prelude::*;

type Fz2U1 = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
type Rule = ProductFusionRule<Fz2U1, SU2FusionRule>;

fn main() -> Result<(), Error> {
    let iterations = std::env::args()
        .nth(1)
        .map(|value| value.parse().expect("iterations must be an integer"))
        .unwrap_or(10_000usize);
    let runtime = Runtime::builder().dense_threads(1).build()?;
    let provider = Rule::new(
        Fz2U1::new(FermionParityFusionRule, U1FusionRule),
        SU2FusionRule,
    );
    let label = |parity: Z2Irrep, charge: i32, twice_spin: usize| {
        ProductSector::new(
            ProductSector::new(parity, U1Irrep::new(charge)),
            SU2Irrep::from_twice_spin(twice_spin),
        )
    };
    let space = GradedSpace::try_new_owned(
        provider,
        [
            (label(Z2Irrep::EVEN, -2, 0), 4),
            (label(Z2Irrep::EVEN, 1, 2), 3),
            (label(Z2Irrep::ODD, -1, 1), 4),
            (label(Z2Irrep::ODD, 2, 3), 2),
        ],
        false,
    )?;
    let lhs = TensorMap::<Rule, Complex64>::rand_with_seed(
        &runtime,
        [&space, &space],
        [&space],
        282_501,
    )?;
    let rhs = TensorMap::<Rule, Complex64>::rand_with_seed(
        &runtime,
        [&space, &space],
        [&space],
        282_502,
    )?;

    let cold_start = Instant::now();
    let cold = lhs.inner(&rhs)?;
    let cold_elapsed = cold_start.elapsed();

    let warm_inner_start = Instant::now();
    let mut inner_checksum = Complex64::new(0.0, 0.0);
    for _ in 0..iterations {
        inner_checksum += black_box(lhs.inner(&rhs)?);
    }
    let warm_inner_elapsed = warm_inner_start.elapsed();

    let warm_norm_start = Instant::now();
    let mut norm_checksum = 0.0;
    for _ in 0..iterations {
        norm_checksum += black_box(lhs.norm()?);
    }
    let warm_norm_elapsed = warm_norm_start.elapsed();

    println!(
        "cold_inner_with_region_init_ns\t{}",
        cold_elapsed.as_nanos()
    );
    println!(
        "warm_inner_ns_per_op\t{:.3}",
        warm_inner_elapsed.as_nanos() as f64 / iterations as f64
    );
    println!(
        "warm_norm_ns_per_op\t{:.3}",
        warm_norm_elapsed.as_nanos() as f64 / iterations as f64
    );
    println!("cold_value\t{}\t{}", cold.re, cold.im);
    println!(
        "warm_inner_checksum\t{}\t{}",
        inner_checksum.re, inner_checksum.im
    );
    println!("warm_norm_checksum\t{norm_checksum}");
    Ok(())
}
