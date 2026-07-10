/// Vector operations required by matrix-free Krylov solvers.
pub trait KrylovVector: Clone {
    /// Return a zero vector with the same shape and storage placement as `self`.
    fn zero_like(&self) -> Self;

    /// Compute `self += alpha * x`.
    fn axpy(&mut self, alpha: f64, x: &Self);

    /// Compute `self *= alpha`.
    fn scale(&mut self, alpha: f64);

    /// Real part of the inner product between `self` and `rhs`.
    fn dot_real(&self, rhs: &Self) -> f64;

    /// Euclidean norm induced by [`KrylovVector::dot_real`].
    #[inline]
    fn norm2(&self) -> f64 {
        self.dot_real(self).sqrt()
    }
}

impl KrylovVector for Vec<f64> {
    fn zero_like(&self) -> Self {
        vec![0.0; self.len()]
    }

    fn axpy(&mut self, alpha: f64, x: &Self) {
        assert_eq!(self.len(), x.len(), "KrylovVector::axpy length mismatch");
        for (lhs, rhs) in self.iter_mut().zip(x) {
            *lhs += alpha * rhs;
        }
    }

    fn scale(&mut self, alpha: f64) {
        for value in self {
            *value *= alpha;
        }
    }

    fn dot_real(&self, rhs: &Self) -> f64 {
        assert_eq!(
            self.len(),
            rhs.len(),
            "KrylovVector::dot_real length mismatch"
        );
        self.iter().zip(rhs).map(|(lhs, rhs)| lhs * rhs).sum()
    }
}
