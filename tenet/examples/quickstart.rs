use tenet::prelude::*;
use tenet_network::tensor;

fn main() -> Result<(), Error> {
    let runtime = Runtime::builder().build()?;
    let space = GradedSpace::try_new_owned(
        U1FusionRule,
        [
            (U1Irrep::new(-1), 1),
            (U1Irrep::new(0), 2),
            (U1Irrep::new(1), 1),
        ],
        false,
    )?;

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

    let c = tensor!([i; k] = a[i; j] * b[j; k])?;
    assert_eq!((c.codomain_rank(), c.domain_rank()), (1, 1));
    let inner = c.inner(&c)?;
    assert_eq!(inner, 50.0);

    println!(
        "result: rank {} <- {}, squared norm = {inner}",
        c.codomain_rank(),
        c.domain_rank()
    );
    Ok(())
}
