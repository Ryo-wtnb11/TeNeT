#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DenseDotConfig {
    lhs_contracting_dims: Vec<usize>,
    rhs_contracting_dims: Vec<usize>,
    lhs_batch_dims: Vec<usize>,
    rhs_batch_dims: Vec<usize>,
}

impl DenseDotConfig {
    pub fn new(
        lhs_contracting_dims: Vec<usize>,
        rhs_contracting_dims: Vec<usize>,
        lhs_batch_dims: Vec<usize>,
        rhs_batch_dims: Vec<usize>,
    ) -> Self {
        Self {
            lhs_contracting_dims,
            rhs_contracting_dims,
            lhs_batch_dims,
            rhs_batch_dims,
        }
    }

    pub fn matmul() -> Self {
        Self::new(vec![1], vec![0], Vec::new(), Vec::new())
    }

    #[inline]
    pub fn lhs_contracting_dims(&self) -> &[usize] {
        &self.lhs_contracting_dims
    }

    #[inline]
    pub fn rhs_contracting_dims(&self) -> &[usize] {
        &self.rhs_contracting_dims
    }

    #[inline]
    pub fn lhs_batch_dims(&self) -> &[usize] {
        &self.lhs_batch_dims
    }

    #[inline]
    pub fn rhs_batch_dims(&self) -> &[usize] {
        &self.rhs_batch_dims
    }
}
