use super::null_space::numerical_rank_and_compact_basis;
use super::*;
use tenet_dense::{DefaultDenseExecutor, DenseRead, DenseWrite};

#[derive(Default)]
struct OwnedSvdSpy {
    inner: DefaultDenseExecutor,
    u_pointer: Option<usize>,
    svd_into_calls: usize,
}

impl DenseExecutor for OwnedSvdSpy {
    fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        let outputs = self.inner.svd(input)?;
        self.u_pointer = Some(
            outputs[0]
                .as_f64_slice()
                .expect("f64 null fixture must return f64 U")
                .as_ptr() as usize,
        );
        Ok(outputs)
    }

    fn svd_into(
        &mut self,
        _: DenseRead<'_>,
        _: DenseWrite<'_>,
        _: DenseWrite<'_>,
        _: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        self.svd_into_calls += 1;
        Err(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op: "svd_into",
            message: "numerical null must consume owned SVD factors".to_string(),
        })
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
        self.inner.dot_general_into(output, lhs, rhs, config)
    }
}

#[test]
fn numerical_null_left_basis_keeps_owned_svd_u() {
    let mut dense = OwnedSvdSpy::default();
    let (rank, u) = numerical_rank_and_compact_basis(
        &mut dense,
        &[1.0_f64, 0.0, 0.0, 0.0, 2.0, 0.0],
        3,
        2,
        FactorSide::Left,
    )
    .unwrap();

    assert_eq!(rank, 2);
    assert_eq!(u.len(), 6);
    assert_eq!(dense.svd_into_calls, 0);
    assert_eq!(u.as_ptr() as usize, dense.u_pointer.unwrap());
}
