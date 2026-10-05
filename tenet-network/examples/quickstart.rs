use tenet::sector::{U1FusionRule, U1Irrep};
use tenet::typed::{Error, GradedSpace, Runtime, TensorMap};
use tenet_network::{Network, TemporaryLabel};

/// One operand's or the output's written labels.
fn labels(names: &[&str]) -> Vec<TemporaryLabel> {
    names.iter().copied().map(TemporaryLabel::from).collect()
}

fn main() -> Result<(), Error> {
    let runtime = Runtime::builder().build()?;
    let space = GradedSpace::try_new(
        std::sync::Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 1),
            (U1Irrep::new(0), 2),
            (U1Irrep::new(1), 1),
        ],
    )?;

    // Fill each allowed charge block. Unequal indices stay zero, so both maps
    // are diagonal.
    let a: TensorMap<U1FusionRule, f64> =
        TensorMap::from_subblock_fn(&runtime, [&space], [&space], |trees, indices| {
            if indices[0] == indices[1] {
                f64::from(2 + trees.coupled().charge())
            } else {
                0.0
            }
        })?;
    let b: TensorMap<U1FusionRule, f64> =
        TensorMap::from_subblock_fn(&runtime, [&space], [&space], |trees, indices| {
            if indices[0] == indices[1] {
                f64::from(2 - trees.coupled().charge())
            } else {
                0.0
            }
        })?;

    // The repeated j is contracted, leaving codomain i and domain k.
    let c = Network::new(
        vec![labels(&["i", "j"]), labels(&["j", "k"])],
        vec![false, false],
        vec![Some(1), Some(1)],
        labels(&["i", "k"]),
        Some(1),
    )?
    .contract(&[&a, &b])?;
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
