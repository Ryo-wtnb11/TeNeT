use tenet::prelude::*;
use tenet_network::tensor;

fn main() -> Result<(), Error> {
    let runtime = Runtime::builder().build()?;
    let space = GradedSpace::try_new(
        U1FusionRule,
        [
            (U1Irrep::new(-1), 1),
            (U1Irrep::new(0), 2),
            (U1Irrep::new(1), 1),
        ],
    )?;

    // Fill each allowed charge block. Unequal indices stay zero, so both maps
    // are diagonal.
    let a: TensorMap<U1FusionRule, f64> =
        TensorMap::from_block_fn(&runtime, [&space], [&space], |trees, indices| {
            if indices[0] == indices[1] {
                f64::from(2 + trees.coupled().charge())
            } else {
                0.0
            }
        })?;
    let b: TensorMap<U1FusionRule, f64> =
        TensorMap::from_block_fn(&runtime, [&space], [&space], |trees, indices| {
            if indices[0] == indices[1] {
                f64::from(2 - trees.coupled().charge())
            } else {
                0.0
            }
        })?;

    // The repeated j is contracted, leaving codomain i and domain k.
    let c = tensor!([i; k] = a[i; j] * b[j; k])?;
    assert_eq!((c.codomain_rank(), c.domain_rank()), (1, 1));
    // A tensor's inner product with itself is its squared norm.
    let inner = c.inner(&c)?;
    assert_eq!(inner, 50.0);

    println!(
        "result: codomain rank = {}, domain rank = {}, squared norm = {inner}",
        c.codomain_rank(),
        c.domain_rank()
    );
    Ok(())
}
