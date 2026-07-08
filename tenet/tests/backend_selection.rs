//! `RuntimeBuilder::with_dense_executor` lets a caller select the CPU
//! linear-algebra backend by injecting a `DenseExecutor` (issue #64). This
//! checks the runtime actually drives the injected executor and that doing so
//! is numerically identical to the faer default.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tenet::dense::{
    DefaultDenseExecutor, DenseDotConfig, DenseError, DenseExecutor, DenseRead, DenseTensor,
    DenseWrite,
};
use tenet::prelude::*;

/// Delegates every dense op to the faer default, counting SVD calls so the test
/// can prove the injected backend is the one the runtime drives. Only the four
/// required `DenseExecutor` methods need forwarding; the rest inherit defaults.
struct SpyExecutor {
    inner: DefaultDenseExecutor,
    svd_calls: Arc<AtomicUsize>,
}

impl DenseExecutor for SpyExecutor {
    fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.svd_calls.fetch_add(1, Ordering::Relaxed);
        self.inner.svd(input)
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
fn injected_dense_executor_is_used_and_preserves_results() {
    let counter = Arc::new(AtomicUsize::new(0));
    let spy = SpyExecutor {
        inner: DefaultDenseExecutor::default(),
        svd_calls: Arc::clone(&counter),
    };
    let rt = Runtime::builder()
        .with_dense_executor(Box::new(spy))
        .build()
        .unwrap();

    let v = Space::u1([(-1, 2), (0, 2), (1, 1)]);
    let t = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v, &v], 99).unwrap();
    let (_, s, _) = t.svd_compact().unwrap();

    assert!(
        counter.load(Ordering::Relaxed) > 0,
        "the injected executor's svd was never called — the runtime is not \
         driving the injected backend"
    );

    // No behavior change: the same seeded tensor on the faer default runtime
    // yields identical singular values.
    let rt_default = Runtime::builder().build().unwrap();
    let t_default =
        Tensor::rand_with_seed(&rt_default, Dtype::F64, [&v, &v], [&v, &v], 99).unwrap();
    let (_, s_default, _) = t_default.svd_compact().unwrap();
    assert_eq!(s.data().len(), s_default.data().len());
    for (a, b) in s.data().iter().zip(s_default.data()) {
        assert!(
            (a - b).abs() <= 1e-12 * (1.0 + a.abs()),
            "singular value differs from the default backend: {a} vs {b}"
        );
    }
}
