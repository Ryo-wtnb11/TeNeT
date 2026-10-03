use super::*;

/// Expert error returned by checked Generic-fusion structural construction.
///
/// The provider source remains typed; core structural errors retain their
/// established variants instead of being flattened into a string envelope.
#[derive(Debug)]
pub enum CheckedGenericStructureError<E> {
    Provider(E),
    Core(CoreError),
}

impl<E> From<CoreError> for CheckedGenericStructureError<E> {
    fn from(error: CoreError) -> Self {
        Self::Core(error)
    }
}

impl<E: fmt::Display> fmt::Display for CheckedGenericStructureError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Provider(error) => error.fmt(formatter),
            Self::Core(error) => error.fmt(formatter),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for CheckedGenericStructureError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Provider(error) => Some(error),
            Self::Core(error) => Some(error),
        }
    }
}

/// Fallible Generic-symbol access error used internally by checked row
/// lowering.  A shape mismatch is TeNeT validation, not a provider failure.
#[doc(hidden)]
#[derive(Debug)]
pub enum CheckedGenericSymbolError<E> {
    Provider(E),
    Shape {
        symbol: &'static str,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    Core(CoreError),
}

impl<E> From<CoreError> for CheckedGenericSymbolError<E> {
    fn from(error: CoreError) -> Self {
        Self::Core(error)
    }
}

impl<E: fmt::Display> fmt::Display for CheckedGenericSymbolError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Provider(error) => error.fmt(formatter),
            Self::Shape {
                symbol,
                expected,
                actual,
            } => write!(
                formatter,
                "{symbol} shape mismatch: expected {expected:?}, got {actual:?}"
            ),
            Self::Core(error) => error.fmt(formatter),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for CheckedGenericSymbolError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Provider(error) => Some(error),
            Self::Core(error) => Some(error),
            Self::Shape { .. } => None,
        }
    }
}

pub(super) fn map_infallible_generic_symbol_error(
    error: CheckedGenericSymbolError<std::convert::Infallible>,
) -> CoreError {
    match error {
        CheckedGenericSymbolError::Provider(never) => match never {},
        CheckedGenericSymbolError::Core(error) => error,
        CheckedGenericSymbolError::Shape { symbol, .. } => CoreError::MalformedFusionTree {
            message: if symbol == "F" {
                "Generic F-symbol shape mismatch"
            } else {
                "Generic symbol shape mismatch"
            },
        },
    }
}

pub(crate) trait GenericFRAccess {
    type Scalar: CategoricalScalar;
    type Error;
    fn fusion_style(&self) -> FusionStyleKind;
    fn braiding_style(&self) -> BraidingStyleKind;
    fn vacuum(&self) -> SectorId;
    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error>;
    fn try_nsymbol(&self, a: SectorId, b: SectorId, c: SectorId) -> Result<usize, Self::Error>;
    fn try_fusion_channels_in_table(
        &self,
        a: SectorId,
        b: SectorId,
    ) -> Result<SectorVec, Self::Error>;
    fn try_f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Result<GenericFArray<Self::Scalar>, Self::Error>;
    fn try_validated_f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Result<GenericFArray<Self::Scalar>, CheckedGenericSymbolError<Self::Error>>;
    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<Self::Scalar>, Self::Error>;
}

pub(crate) trait GenericRigidAccess: GenericFRAccess {
    fn try_sqrt_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error>;
    fn try_inv_sqrt_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error>;
    fn try_frobenius_schur_phase_scalar(
        &self,
        sector: SectorId,
    ) -> Result<Self::Scalar, Self::Error>;
    fn try_b_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<Self::Scalar>, CheckedGenericSymbolError<Self::Error>>;
    #[allow(dead_code)]
    fn try_a_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<Self::Scalar>, CheckedGenericSymbolError<Self::Error>>;
}

pub(crate) struct InfallibleGenericFR<'a, R>(pub(crate) &'a R);

impl<R> GenericFRAccess for InfallibleGenericFR<'_, R>
where
    R: GenericFusionSymbols,
    R::Scalar: CategoricalScalar,
{
    type Scalar = R::Scalar;
    type Error = std::convert::Infallible;
    fn fusion_style(&self) -> FusionStyleKind {
        self.0.fusion_style()
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        self.0.braiding_style()
    }
    fn vacuum(&self) -> SectorId {
        self.0.vacuum()
    }
    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        Ok(self.0.dual(sector))
    }
    fn try_nsymbol(&self, a: SectorId, b: SectorId, c: SectorId) -> Result<usize, Self::Error> {
        Ok(self.0.nsymbol(a, b, c))
    }
    fn try_fusion_channels_in_table(
        &self,
        a: SectorId,
        b: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        Ok(self.0.fusion_channels(a, b))
    }
    fn try_f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Result<GenericFArray<Self::Scalar>, Self::Error> {
        Ok(self.0.f_symbol_generic(a, b, c, d, e, f))
    }
    fn try_validated_f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Result<GenericFArray<Self::Scalar>, CheckedGenericSymbolError<Self::Error>> {
        Ok(self.0.f_symbol_generic(a, b, c, d, e, f))
    }
    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<Self::Scalar>, Self::Error> {
        Ok(self.0.r_symbol_generic(a, b, c))
    }
}

impl<P: CheckedGenericRigidSymbols> GenericFRAccess for P {
    type Scalar = P::Scalar;
    type Error = P::Error;
    fn fusion_style(&self) -> FusionStyleKind {
        CheckedGenericFusion::fusion_style(self)
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        CheckedGenericFusion::braiding_style(self)
    }
    fn vacuum(&self) -> SectorId {
        CheckedGenericFusion::vacuum(self)
    }
    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        CheckedGenericFusion::try_dual(self, sector)
    }
    fn try_nsymbol(&self, a: SectorId, b: SectorId, c: SectorId) -> Result<usize, Self::Error> {
        CheckedGenericFusion::try_nsymbol(self, a, b, c)
    }
    fn try_fusion_channels_in_table(
        &self,
        a: SectorId,
        b: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        CheckedGenericFusion::try_fusion_channels_in_table(self, a, b)
    }
    fn try_f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Result<GenericFArray<Self::Scalar>, Self::Error> {
        CheckedGenericRigidSymbols::try_f_symbol_generic(self, a, b, c, d, e, f)
    }
    fn try_validated_f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Result<GenericFArray<Self::Scalar>, CheckedGenericSymbolError<Self::Error>> {
        checked_generic_f_symbol(self, a, b, c, d, e, f)
    }
    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<Self::Scalar>, Self::Error> {
        CheckedGenericRigidSymbols::try_r_symbol_generic(self, a, b, c)
    }
}

pub(super) struct InfallibleGenericRigid<'a, R>(pub(super) &'a R);

impl<R> GenericFRAccess for InfallibleGenericRigid<'_, R>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    type Scalar = R::Scalar;
    type Error = std::convert::Infallible;
    fn fusion_style(&self) -> FusionStyleKind {
        self.0.fusion_style()
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        self.0.braiding_style()
    }
    fn vacuum(&self) -> SectorId {
        self.0.vacuum()
    }
    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        Ok(self.0.dual(sector))
    }
    fn try_nsymbol(&self, a: SectorId, b: SectorId, c: SectorId) -> Result<usize, Self::Error> {
        Ok(self.0.nsymbol(a, b, c))
    }
    fn try_fusion_channels_in_table(
        &self,
        a: SectorId,
        b: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        Ok(self.0.fusion_channels(a, b))
    }
    fn try_f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Result<GenericFArray<Self::Scalar>, Self::Error> {
        Ok(self.0.f_symbol_generic(a, b, c, d, e, f))
    }
    fn try_validated_f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Result<GenericFArray<Self::Scalar>, CheckedGenericSymbolError<Self::Error>> {
        Ok(self.0.f_symbol_generic(a, b, c, d, e, f))
    }
    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<Self::Scalar>, Self::Error> {
        Ok(self.0.r_symbol_generic(a, b, c))
    }
}

impl<R> GenericRigidAccess for InfallibleGenericRigid<'_, R>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    fn try_sqrt_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        Ok(self.0.sqrt_dim_scalar(sector))
    }

    fn try_inv_sqrt_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        Ok(self.0.inv_sqrt_dim_scalar(sector))
    }

    fn try_frobenius_schur_phase_scalar(
        &self,
        sector: SectorId,
    ) -> Result<Self::Scalar, Self::Error> {
        Ok(self.0.frobenius_schur_phase_scalar(sector))
    }

    fn try_b_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<Self::Scalar>, CheckedGenericSymbolError<Self::Error>> {
        Ok(self.0.b_symbol_generic(a, b, c))
    }

    fn try_a_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<Self::Scalar>, CheckedGenericSymbolError<Self::Error>> {
        Ok(self.0.a_symbol_generic(a, b, c))
    }
}

impl<P: CheckedGenericRigidSymbols> GenericRigidAccess for P {
    fn try_sqrt_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        CheckedGenericRigidSymbols::try_sqrt_dim_scalar(self, sector)
    }

    fn try_inv_sqrt_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        CheckedGenericRigidSymbols::try_inv_sqrt_dim_scalar(self, sector)
    }

    fn try_frobenius_schur_phase_scalar(
        &self,
        sector: SectorId,
    ) -> Result<Self::Scalar, Self::Error> {
        CheckedGenericRigidSymbols::try_frobenius_schur_phase_scalar(self, sector)
    }

    fn try_b_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<Self::Scalar>, CheckedGenericSymbolError<Self::Error>> {
        checked_generic_b_symbol(self, a, b, c)
    }

    fn try_a_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<Self::Scalar>, CheckedGenericSymbolError<Self::Error>> {
        checked_generic_a_symbol(self, a, b, c)
    }
}

/// Validate a Generic fusion-tree pair without requiring the infallible
/// [`FusionRule`] contract.
pub fn validate_generic_fusion_tree_pair_checked<C>(
    rule: &C,
    tree_pair: &FusionTreePairKey,
) -> Result<(), CheckedGenericStructureError<C::Error>>
where
    C: CheckedGenericFusion,
{
    if !rule.fusion_style().has_multiplicity() {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: rule.fusion_style(),
        }
        .into());
    }
    for tree in [tree_pair.codomain_tree(), tree_pair.domain_tree()] {
        validate_fusion_tree_key_shape(tree)?;
        validate_fusion_tree_checked_after_shape(
            tree,
            rule.vacuum(),
            |left, right| {
                rule.try_fusion_channels(left, right)
                    .map(drop)
                    .map_err(CheckedGenericStructureError::Provider)
            },
            |left, right, coupled| {
                rule.try_nsymbol(left, right, coupled)
                    .map_err(CheckedGenericStructureError::Provider)
            },
        )?;
    }
    validate_fusion_tree_pair_coupled(tree_pair.codomain_tree(), tree_pair.domain_tree())?;
    Ok(())
}

pub(super) fn map_checked_generic_structure_error<E>(
    error: CheckedGenericStructureError<E>,
) -> CheckedGenericSymbolError<E> {
    match error {
        CheckedGenericStructureError::Provider(error) => CheckedGenericSymbolError::Provider(error),
        CheckedGenericStructureError::Core(error) => CheckedGenericSymbolError::Core(error),
    }
}

pub(super) fn checked_generic_f_symbol<C>(
    rule: &C,
    a: SectorId,
    b: SectorId,
    c: SectorId,
    d: SectorId,
    e: SectorId,
    f: SectorId,
) -> Result<GenericFArray<C::Scalar>, CheckedGenericSymbolError<C::Error>>
where
    C: GenericFRAccess,
{
    let expected = [
        rule.try_nsymbol(a, b, e),
        rule.try_nsymbol(e, c, d),
        rule.try_nsymbol(b, c, f),
        rule.try_nsymbol(a, f, d),
    ]
    .into_iter()
    .collect::<Result<Vec<_>, _>>()
    .map_err(CheckedGenericSymbolError::Provider)?;
    let symbol = rule
        .try_f_symbol_generic(a, b, c, d, e, f)
        .map_err(CheckedGenericSymbolError::Provider)?;
    let (mu, nu, kappa, lambda) = symbol.shape();
    let actual = vec![mu, nu, kappa, lambda];
    if actual != expected {
        return Err(CheckedGenericSymbolError::Shape {
            symbol: "F",
            expected,
            actual,
        });
    }
    Ok(symbol)
}

pub(super) fn checked_generic_r_symbol<C>(
    rule: &C,
    a: SectorId,
    b: SectorId,
    c: SectorId,
) -> Result<GenericRMatrix<C::Scalar>, CheckedGenericSymbolError<C::Error>>
where
    C: GenericFRAccess,
{
    let expected = [rule.try_nsymbol(a, b, c), rule.try_nsymbol(b, a, c)]
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .map_err(CheckedGenericSymbolError::Provider)?;
    let symbol = rule
        .try_r_symbol_generic(a, b, c)
        .map_err(CheckedGenericSymbolError::Provider)?;
    let (rows, cols) = symbol.shape();
    let actual = vec![rows, cols];
    if actual != expected {
        return Err(CheckedGenericSymbolError::Shape {
            symbol: "R",
            expected,
            actual,
        });
    }
    Ok(symbol)
}

fn checked_generic_b_symbol<C>(
    rule: &C,
    a: SectorId,
    b: SectorId,
    c: SectorId,
) -> Result<GenericRMatrix<C::Scalar>, CheckedGenericSymbolError<C::Error>>
where
    C: CheckedGenericRigidSymbols,
{
    let rows = rule
        .try_nsymbol(a, b, c)
        .map_err(CheckedGenericSymbolError::Provider)?;
    let dual_b = rule
        .try_dual(b)
        .map_err(CheckedGenericSymbolError::Provider)?;
    let cols = rule
        .try_nsymbol(c, dual_b, a)
        .map_err(CheckedGenericSymbolError::Provider)?;
    let vacuum = rule.vacuum();
    let f = checked_generic_f_symbol(rule, a, b, dual_b, a, c, vacuum)?;
    let factor = rule
        .try_sqrt_dim_scalar(a)
        .map_err(CheckedGenericSymbolError::Provider)?
        * rule
            .try_sqrt_dim_scalar(b)
            .map_err(CheckedGenericSymbolError::Provider)?
        * rule
            .try_inv_sqrt_dim_scalar(c)
            .map_err(CheckedGenericSymbolError::Provider)?;
    let mut data = Vec::with_capacity(rows * cols);
    for mu in 0..rows {
        for nu in 0..cols {
            data.push(factor.clone() * f.get(mu, nu, 0, 0).clone());
        }
    }
    Ok(GenericRMatrix::new(data, rows, cols))
}

fn checked_generic_a_symbol<C>(
    rule: &C,
    a: SectorId,
    b: SectorId,
    c: SectorId,
) -> Result<GenericRMatrix<C::Scalar>, CheckedGenericSymbolError<C::Error>>
where
    C: CheckedGenericRigidSymbols,
{
    let rows = rule
        .try_nsymbol(a, b, c)
        .map_err(CheckedGenericSymbolError::Provider)?;
    let dual_a = rule
        .try_dual(a)
        .map_err(CheckedGenericSymbolError::Provider)?;
    let cols = rule
        .try_nsymbol(dual_a, c, b)
        .map_err(CheckedGenericSymbolError::Provider)?;
    let vacuum = rule.vacuum();
    let f = checked_generic_f_symbol(rule, dual_a, a, b, b, vacuum, c)?;
    let factor = rule
        .try_sqrt_dim_scalar(a)
        .map_err(CheckedGenericSymbolError::Provider)?
        * rule
            .try_sqrt_dim_scalar(b)
            .map_err(CheckedGenericSymbolError::Provider)?
        * rule
            .try_inv_sqrt_dim_scalar(c)
            .map_err(CheckedGenericSymbolError::Provider)?;
    let fs = rule
        .try_frobenius_schur_phase_scalar(a)
        .map_err(CheckedGenericSymbolError::Provider)?;
    let mut data = Vec::with_capacity(rows * cols);
    for kappa in 0..rows {
        for lambda in 0..cols {
            data.push(factor.clone() * (fs.clone() * f.get(0, 0, kappa, lambda).clone()).conj());
        }
    }
    Ok(GenericRMatrix::new(data, rows, cols))
}
