/// Matrix-free linear operator.
pub trait LinearOperator<V> {
    /// Return `A * x`.
    fn apply(&self, x: &V) -> V;
}

impl<V, F> LinearOperator<V> for F
where
    F: Fn(&V) -> V,
{
    #[inline]
    fn apply(&self, x: &V) -> V {
        self(x)
    }
}
