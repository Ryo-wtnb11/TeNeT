#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProductSpace<const N: usize> {
    dims: [usize; N],
    dim: usize,
}

impl<const N: usize> ProductSpace<N> {
    pub fn new(dims: [usize; N]) -> Result<Self, CoreError> {
        let dim = checked_product(&dims)?;
        Ok(Self { dims, dim })
    }

    #[inline]
    pub fn dims(&self) -> &[usize; N] {
        &self.dims
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorMapSpace<const NOUT: usize, const NIN: usize> {
    codomain: ProductSpace<NOUT>,
    domain: ProductSpace<NIN>,
    dims: DimVec,
    dense_dim: usize,
}

impl<const NOUT: usize, const NIN: usize> TensorMapSpace<NOUT, NIN> {
    pub fn new(codomain: ProductSpace<NOUT>, domain: ProductSpace<NIN>) -> Result<Self, CoreError> {
        let dense_dim = codomain
            .dim()
            .checked_mul(domain.dim())
            .ok_or(CoreError::ElementCountOverflow)?;
        let mut dims = DimVec::with_capacity(NOUT + NIN);
        dims.extend_from_slice(codomain.dims());
        dims.extend_from_slice(domain.dims());
        Ok(Self {
            codomain,
            domain,
            dims,
            dense_dim,
        })
    }

    /// Builds a dense tensor-map space from codomain and domain dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenet_core::TensorMapSpace;
    ///
    /// let space = TensorMapSpace::<2, 1>::from_dims([2, 3], [4]).unwrap();
    /// assert_eq!(space.dims(), &[2, 3, 4]);
    /// assert_eq!(space.dense_dim(), 24);
    /// ```
    pub fn from_dims(codomain: [usize; NOUT], domain: [usize; NIN]) -> Result<Self, CoreError> {
        Self::new(ProductSpace::new(codomain)?, ProductSpace::new(domain)?)
    }

    #[inline]
    pub fn codomain(&self) -> &ProductSpace<NOUT> {
        &self.codomain
    }

    #[inline]
    pub fn domain(&self) -> &ProductSpace<NIN> {
        &self.domain
    }

    #[inline]
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    #[inline]
    pub fn dense_dim(&self) -> usize {
        self.dense_dim
    }
}
