use std::hint::black_box;
use std::time::Instant;

use tenet::prelude::*;

fn main() -> Result<(), Error> {
    let iterations = std::env::args()
        .nth(1)
        .map(|value| value.parse().expect("iterations must be an integer"))
        .unwrap_or(10_000usize);
    let runtime = Runtime::builder().dense_threads(1).build()?;
    let space = Space::fz2_u1_su2([
        ((0, -2, 0), 4),
        ((0, 1, 2), 3),
        ((1, -1, 1), 4),
        ((1, 2, 3), 2),
    ])?;
    let lhs = Tensor::rand_with_seed(&runtime, Dtype::C64, [&space, &space], [&space], 282_501)?;
    let rhs = Tensor::rand_with_seed(&runtime, Dtype::C64, [&space, &space], [&space], 282_502)?;

    let cold_start = Instant::now();
    let cold = lhs.inner(&rhs)?.to_c64();
    let cold_elapsed = cold_start.elapsed();

    let warm_inner_start = Instant::now();
    let mut inner_checksum = Complex64::new(0.0, 0.0);
    for _ in 0..iterations {
        inner_checksum += black_box(lhs.inner(&rhs)?.to_c64());
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
