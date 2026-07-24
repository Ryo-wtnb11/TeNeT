use core::fmt;
use core::marker::PhantomData;
use core::ops::{Add, Mul};

use num_complex::Complex64;

use crate::{BraidingStyleKind, FusionStyleKind, RuleIdentity, SectorId, SectorVec};

/// Classification of a leg tuple's coupled-sector candidates for a possibly
/// table-bounded rule (Stage B3b Option A). For unbounded rules every
/// candidate is `clean`; for the bounded SU(3) table:
/// * `clean`: every fusion tree ending in this sector stays inside the table —
///   enumeration is exactly the full-SU(3) tree set;
/// * `tainted`: some full-SU(3) tree for this sector passes through an
///   out-of-table inner line — the table cannot enumerate (or recouple) the
///   complete set, so constructing this sector must be an error, NEVER a
///   silently truncated block;
/// * `out_of_table`: display labels of coupled-sector candidates that escaped
///   the table entirely (they cannot even be named by a dense `SectorId`);
/// * `poisoned`: the fold left the one-hop frontier shell, so the split into
///   clean/tainted is unknown — every sector must be treated as tainted
///   (`clean` is emptied by the producer when this fires).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CoupledSectorFold {
    pub clean: Vec<SectorId>,
    pub tainted: Vec<SectorId>,
    pub out_of_table: Vec<String>,
    pub poisoned: bool,
}

impl CoupledSectorFold {
    /// Whether the fold proved the candidate set complete and in-table.
    pub fn is_fully_clean(&self) -> bool {
        self.tainted.is_empty() && self.out_of_table.is_empty() && !self.poisoned
    }
}

pub trait FusionRule: 'static {
    /// Stable semantic identity of this provider.
    ///
    /// Equal identities require immutable equivalence of every operation that
    /// can affect fusion-space structure or numerical recoupling, including
    /// fusion/braiding styles, vacuum, fusion channels, duals, F/R symbols,
    /// dimensions, twists, and bends.
    /// Returning one identity for semantically different providers violates
    /// this trait contract.
    ///
    /// Why not `TypeId` or the provider address: one Rust type may hold
    /// different tables, while distinct allocations may represent the same
    /// immutable rule and must be safe to share in semantic caches.
    fn rule_identity(&self) -> RuleIdentity;

    fn fusion_style(&self) -> FusionStyleKind;

    fn braiding_style(&self) -> BraidingStyleKind;

    fn vacuum(&self) -> SectorId;

    fn supports_unitary_braid_dagger(&self) -> bool {
        false
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        sector
    }

    /// Fusion channels of `left ⊗ right`. Returns a `SectorVec` so the common
    /// small result stays stack-inline — this is called millions of times in
    /// the cold recoupling build, and a heap `Vec` per call was ~5% of all
    /// cold-path allocations.
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec;

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        usize::from(self.fusion_channels(left, right).contains(&coupled))
    }

    /// The representable channels of `left ⊗ right` — identical to
    /// [`Self::fusion_channels`] for unbounded rules (the default). A
    /// table-bounded rule (SU(3), Stage B3b) overrides this to return the
    /// in-table channels of an ESCAPING pair instead of panicking.
    ///
    /// Safety contract: callers may only rely on this being the complete
    /// channel list where out-of-table branches provably contribute nothing —
    /// i.e. on trees of sectors the [`Self::coupled_sector_fold`] classified
    /// `clean` (a clean sector by definition has no tree through an
    /// out-of-table line, so skipping frontier channels drops only
    /// provably-dead branches).
    fn fusion_channels_in_table(&self, left: SectorId, right: SectorId) -> SectorVec {
        self.fusion_channels(left, right)
    }

    /// Folds `effective[0] ⊗ effective[1] ⊗ …` and classifies every coupled
    /// candidate (see [`CoupledSectorFold`]). Default: the plain unbounded
    /// fold — everything clean, nothing escapes. Bounded rules override.
    fn coupled_sector_fold(&self, effective: &[SectorId]) -> CoupledSectorFold {
        let mut acc: Vec<SectorId> = match effective.first() {
            None => vec![self.vacuum()],
            Some(&first) => vec![first],
        };
        for &last in effective.iter().skip(1) {
            acc = acc
                .iter()
                .flat_map(|&front| self.fusion_channels(front, last))
                .collect();
            acc.sort_unstable();
            acc.dedup();
        }
        acc.sort_unstable();
        acc.dedup();
        CoupledSectorFold {
            clean: acc,
            ..CoupledSectorFold::default()
        }
    }
}

#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FusionAlgebraError {
    /// An input ID does not name a sector in the rule's representable domain.
    InvalidSector {
        sector: SectorId,
    },
    U1DualOverflow {
        charge: i32,
    },
    U1FusionOverflow {
        left: i32,
        right: i32,
    },
    /// Both inputs are valid, but at least one output channel cannot be represented.
    FusionNotRepresentable {
        left: SectorId,
        right: SectorId,
    },
    ProductCodec(ProductSectorCodecError),
    MultiplicityOverflow {
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    },
}

impl fmt::Display for FusionAlgebraError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSector { sector } => write!(formatter, "invalid fusion sector {sector:?}"),
            Self::U1DualOverflow { charge } => {
                write!(
                    formatter,
                    "U(1) dual charge -({charge}) is not representable"
                )
            }
            Self::U1FusionOverflow { left, right } => write!(
                formatter,
                "U(1) fusion charge {left} + {right} is not representable"
            ),
            Self::FusionNotRepresentable { left, right } => write!(
                formatter,
                "fusion output for {left:?} x {right:?} is not representable"
            ),
            Self::ProductCodec(error) => error.fmt(formatter),
            Self::MultiplicityOverflow {
                left,
                right,
                coupled,
            } => write!(
                formatter,
                "fusion multiplicity overflows usize for {left:?} x {right:?} -> {coupled:?}"
            ),
        }
    }
}

impl std::error::Error for FusionAlgebraError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ProductCodec(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ProductSectorCodecError> for FusionAlgebraError {
    fn from(error: ProductSectorCodecError) -> Self {
        Self::ProductCodec(error)
    }
}

/// Checked companion for finite or encoded fusion algebras.
///
/// The infallible [`FusionRule`] methods remain the validated expert hot path.
pub trait CheckedFusionAlgebra: FusionRule {
    /// Return the exact dual when both the input and output are representable
    /// by this provider; otherwise return the exact representation failure.
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError>;

    /// Return the complete mathematical channel set for two representable
    /// inputs.
    ///
    /// On success this is exactly [`FusionRule::fusion_channels`], and every
    /// returned sector is representable by this provider. Failure preserves
    /// the exact cause for an invalid input or an unrepresentable generated
    /// channel.
    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError>;

    /// Return the exact multiplicity for three representable sectors.
    ///
    /// Zero means the coupled sector is mathematically absent from the fusion
    /// product. Failure preserves the exact representation error for any
    /// input sector.
    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError>;
}

pub trait MultiplicityFreeFusionRule: FusionRule {}

pub trait MultiplicityFreeFusionSymbols: MultiplicityFreeFusionRule {
    // Send + Sync because cached recoupling coefficients are shared across
    // tree-transform replay workers (TensorKit sectorscalartype parity: the
    // concrete scalar is a plain number type).
    type Scalar: Clone + Send + Sync;

    fn scalar_one(&self) -> Self::Scalar;

    fn scalar_conj(&self, value: Self::Scalar) -> Self::Scalar;

    /// Whether every allowed associator coefficient in this provider's
    /// current gauge is exactly the scalar unit.
    ///
    /// TensorKit's direct Unique + SymmetricBraiding permutation lowering
    /// relies on this stronger property, not symmetric braiding alone.
    /// Defaulting to false keeps custom providers on the gauge-general Artin
    /// path unless they explicitly certify the invariant.
    fn has_trivial_associator_gauge(&self) -> bool {
        false
    }

    fn f_symbol_scalar(
        &self,
        left: SectorId,
        middle: SectorId,
        right: SectorId,
        coupled: SectorId,
        left_coupled: SectorId,
        right_coupled: SectorId,
    ) -> Self::Scalar;

    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar;
}

// `Sync` because the tree-transform plan compile computes recoupling rows
// for independent source trees in parallel, sharing the rule across workers
// (TensorKit threaded transformer construction parity: sector types are
// plain shareable data). All rule implementations are plain data.
pub trait MultiplicityFreeRigidSymbols: MultiplicityFreeFusionSymbols + Sync {
    fn dim_scalar(&self, sector: SectorId) -> Self::Scalar;

    fn inv_dim_scalar(&self, sector: SectorId) -> Self::Scalar;

    fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar;

    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar;

    fn twist_scalar(&self, sector: SectorId) -> Self::Scalar;

    fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> Self::Scalar;

    fn a_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar
    where
        Self::Scalar: Mul<Output = Self::Scalar>,
    {
        let factor = self.sqrt_dim_scalar(left)
            * self.sqrt_dim_scalar(right)
            * self.inv_sqrt_dim_scalar(coupled);
        let symbol = self.frobenius_schur_phase_scalar(left)
            * self.f_symbol_scalar(self.dual(left), left, right, right, self.vacuum(), coupled);
        factor * self.scalar_conj(symbol)
    }

    fn b_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar
    where
        Self::Scalar: Mul<Output = Self::Scalar>,
    {
        self.sqrt_dim_scalar(left)
            * self.sqrt_dim_scalar(right)
            * self.inv_sqrt_dim_scalar(coupled)
            * self.f_symbol_scalar(left, right, self.dual(right), left, coupled, self.vacuum())
    }
}

/// Dense rank-4 F-symbol block for `FusionStyleKind::Generic` (outer
/// multiplicity) rules: `F(a,b,c,d,e,f)[mu,nu,kappa,lambda]` with shape
/// `(N(a,b,e), N(e,c,d), N(b,c,f), N(a,f,d))` — TensorKit `GenericFusion`
/// convention (`sectors.jl` Fsymbol). Row-major over `[mu,nu,kappa,lambda]`.
///
/// Deliberately a plain `Vec<Scalar>` + shape tuple, not an ndarray-style
/// type: Stage A only needs this to type-check and hold toy-rule test data,
/// nobody indexes it on a hot path yet (that lands with the Stage B recouple
/// wrapper). Reaching for a real N-d array crate now would be solving a
/// problem Stage A doesn't have.
#[derive(Clone, Debug)]
pub struct GenericFArray<Scalar> {
    data: Vec<Scalar>,
    shape: (usize, usize, usize, usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SymbolShapeError {
    pub actual_len: usize,
    pub expected_len: Option<usize>,
}

impl fmt::Display for SymbolShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.expected_len {
            Some(expected) => write!(
                f,
                "symbol data length {} does not match shape product {expected}",
                self.actual_len
            ),
            None => write!(f, "symbol shape product overflows usize"),
        }
    }
}

impl std::error::Error for SymbolShapeError {}

impl<Scalar> GenericFArray<Scalar> {
    pub fn new(data: Vec<Scalar>, shape: (usize, usize, usize, usize)) -> Self {
        Self::try_new(data, shape).unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_new(
        data: Vec<Scalar>,
        shape: (usize, usize, usize, usize),
    ) -> Result<Self, SymbolShapeError> {
        let (n_mu, n_nu, n_kappa, n_lambda) = shape;
        let expected_len = n_mu
            .checked_mul(n_nu)
            .and_then(|n| n.checked_mul(n_kappa))
            .and_then(|n| n.checked_mul(n_lambda));
        if expected_len != Some(data.len()) {
            return Err(SymbolShapeError {
                actual_len: data.len(),
                expected_len,
            });
        }
        Ok(Self { data, shape })
    }

    #[inline]
    pub fn shape(&self) -> (usize, usize, usize, usize) {
        self.shape
    }

    #[inline]
    pub fn data(&self) -> &[Scalar] {
        &self.data
    }

    /// `F[mu,nu,kappa,lambda]`, row-major over the shape tuple.
    pub fn get(&self, mu: usize, nu: usize, kappa: usize, lambda: usize) -> &Scalar {
        let (_, n_nu, n_kappa, n_lambda) = self.shape;
        let idx = ((mu * n_nu + nu) * n_kappa + kappa) * n_lambda + lambda;
        &self.data[idx]
    }
}

/// Dense R-symbol matrix for `FusionStyleKind::Generic` rules:
/// `R(a,b,c)` is `N(a,b,c) x N(b,a,c)`, row-major.
#[derive(Clone, Debug)]
pub struct GenericRMatrix<Scalar> {
    data: Vec<Scalar>,
    rows: usize,
    cols: usize,
}

impl<Scalar> GenericRMatrix<Scalar> {
    pub fn new(data: Vec<Scalar>, rows: usize, cols: usize) -> Self {
        Self::try_new(data, rows, cols).unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_new(data: Vec<Scalar>, rows: usize, cols: usize) -> Result<Self, SymbolShapeError> {
        let expected_len = rows.checked_mul(cols);
        if expected_len != Some(data.len()) {
            return Err(SymbolShapeError {
                actual_len: data.len(),
                expected_len,
            });
        }
        Ok(Self { data, rows, cols })
    }

    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    #[inline]
    pub fn data(&self) -> &[Scalar] {
        &self.data
    }

    pub fn get(&self, row: usize, col: usize) -> &Scalar {
        &self.data[row * self.cols + col]
    }
}

/// Outer-multiplicity ("Generic" fusion, TensorKit `FusionStyle` = `GenericFusion`)
/// sibling of [`MultiplicityFreeFusionSymbols`]. Where the multiplicity-free
/// trait returns a bare `Scalar` per (a,b,c,...) because `nsymbol` is always
/// 0 or 1, this trait returns a dense rank-4 array / matrix because
/// `nsymbol` can exceed 1 (SU(3), SO(N>=7), Sp(N), ...).
///
/// Stage A only: this trait is defined so the shape of the eventual
/// recoupling API type-checks against a toy rule in tests. Nobody implements
/// it outside `tests.rs` yet, and nothing in the recoupling engine consumes
/// it — that wiring (the recouple wrapper, `UnsupportedFusionStyle` guard
/// removal) is explicitly deferred to Stage B pending review of this diff.
pub trait GenericFusionSymbols: FusionRule {
    type Scalar: Clone + Send + Sync;

    fn f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> GenericFArray<Self::Scalar>;

    fn r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> GenericRMatrix<Self::Scalar>;
}

/// Scalar arithmetic the Generic-fusion (outer-multiplicity) Artin braid needs
/// when summing the `R · F̄ · R̄` inner-index contraction
/// (`braiding_manipulations.jl:181-182`:
/// `coeff += Rmat1[ν,ρ] * conj(Fmat[κ,λ,μ,ρ]) * conj(Rmat2[σ,κ])`).
///
/// [`GenericFusionSymbols`] deliberately fixes only `type Scalar: Clone + Send +
/// Sync` — Stage A never *computed* with the scalar, it only stored toy F/R
/// blocks — so the braid layer needs `conj` / `+` / `*` / `zero` / `one` /
/// `iszero` as an extra capability. Expressing that as a bound on `R::Scalar`
/// here (a NEW trait, implemented for the concrete scalar types) keeps the
/// Stage A `GenericFusionSymbols` trait byte-for-byte untouched (pure
/// addition), while mirroring the `scalar_one` / `scalar_conj` that the
/// multiplicity-free [`MultiplicityFreeFusionSymbols`] carries on the rule.
/// `Add`/`Mul` are supertraits so the braid can use the `+`/`*` operators
/// exactly as TensorKit writes them.
pub trait GenericBraidScalar: Clone + Add<Output = Self> + Mul<Output = Self> {
    /// Additive identity — the `coeff = zero(oneT)` accumulator seed at
    /// `braiding_manipulations.jl:179`.
    fn braid_zero() -> Self;

    /// Multiplicative identity — the `oneT` unit-braid / seed coefficient
    /// (`braiding_manipulations.jl:96`, `:117`).
    fn braid_one() -> Self;

    /// Complex conjugation — TensorKit's `conj(...)` and the matrix adjoint
    /// `'` (`braiding_manipulations.jl:139`, `:172-173`, `:181-182`). Real for
    /// real scalars.
    fn braid_conj(&self) -> Self;

    /// Whether this coefficient is exactly zero — the `iszero(R) && continue`
    /// / `iszero(coeff) && continue` prune (`braiding_manipulations.jl:142`,
    /// `:184`). Exact compare mirrors Julia's `iszero`.
    fn braid_is_zero(&self) -> bool;
}

impl GenericBraidScalar for f64 {
    fn braid_zero() -> Self {
        0.0
    }

    fn braid_one() -> Self {
        1.0
    }

    fn braid_conj(&self) -> Self {
        *self
    }

    fn braid_is_zero(&self) -> bool {
        *self == 0.0
    }
}

impl GenericBraidScalar for Complex64 {
    fn braid_zero() -> Self {
        Complex64::new(0.0, 0.0)
    }

    fn braid_one() -> Self {
        Complex64::new(1.0, 0.0)
    }

    fn braid_conj(&self) -> Self {
        Complex64::conj(self)
    }

    fn braid_is_zero(&self) -> bool {
        self.re == 0.0 && self.im == 0.0
    }
}

/// Outer-multiplicity ("Generic" fusion) sibling of
/// [`MultiplicityFreeRigidSymbols`]: the rigidity/pivotal data (quantum
/// dimensions, Frobenius–Schur phase, A/B moves) for a rule whose `nsymbol`
/// can exceed 1. Where the multiplicity-free trait's `a_symbol_scalar` /
/// `b_symbol_scalar` return a bare `Scalar`, here the A/B symbols are
/// `N × N` matrices ([`GenericRMatrix`]), mirroring TensorKitSectors'
/// `Asymbol` / `Bsymbol` for `GenericFusion` (`sectors.jl` v0.3.6).
///
/// Bend/repartition only ever multiply, conjugate and zero-test these
/// coefficients, so the associated scalar carries only the
/// [`GenericBraidScalar`] capability (the same bound the Generic Artin braid
/// uses) — no separate `scalar_one`/`scalar_conj` on the rule.
///
/// `frobenius_schur_phase_scalar` stays a bare `Scalar` (not a matrix): the
/// FS phase is `sign(F^{a ā a}_a[1])` and for *any* fusion style the relevant
/// `F` block has all its `N`-labels forced to 1 by the pivotal axioms, so it
/// is a single number even in the Generic case
/// (TensorKitSectors `sectors.jl:463-468`, `frobenius_schur_phase_from_Fsymbol`).
///
/// `a_symbol_generic` is unused by Stage B2a (bend/repartition need only `B`);
/// it is defined here as the natural sibling that the A-move layer (fold /
/// cycle, Stage B2b) will consume, and is kept honest by a TK oracle test.
pub trait GenericRigidSymbols: GenericFusionSymbols
where
    Self::Scalar: GenericBraidScalar,
{
    /// `√dim(sector)` — TensorKitSectors `sqrtdim` (`sectors.jl:440`).
    fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar;

    /// `1/√dim(sector)` — TensorKitSectors `invsqrtdim` (`sectors.jl:441`).
    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar;

    /// Frobenius–Schur phase `κ_sector` — bare scalar (`sectors.jl:463-468`).
    fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> Self::Scalar;

    /// `B^{ab}_c` as an `N(a,b,c) × N(c, dual(b), a)` matrix, row = input
    /// splitting vertex `μ`, column = output fusion vertex `ν`.
    ///
    /// Verbatim mirror of TensorKitSectors `Bsymbol_from_Fsymbol`
    /// (`sectors.jl:543-551`):
    /// `reshape(√dim(a)·√dim(b)·(1/√dim(c)) · F(a,b,dual(b),a,c,unit),
    ///  (N(a,b,c), N(c,dual(b),a)))`.
    /// The reshape drops the trailing two `F` axes because
    /// `N(b, dual(b), unit) == N(a, unit, a) == 1` (rigidity + unit axioms),
    /// so `B[μ,ν] = F[μ,ν,0,0]`.
    fn b_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> GenericRMatrix<Self::Scalar>
    where
        Self::Scalar: GenericBraidScalar,
    {
        let rows = self.nsymbol(a, b, c);
        let cols = self.nsymbol(c, self.dual(b), a);
        // F(a, b, dual(b), a, c, rightunit(a)); `vacuum()` is the (right)unit.
        let f = self.f_symbol_generic(a, b, self.dual(b), a, c, self.vacuum());
        let factor =
            self.sqrt_dim_scalar(a) * self.sqrt_dim_scalar(b) * self.inv_sqrt_dim_scalar(c);
        let mut data = Vec::with_capacity(rows * cols);
        for mu in 0..rows {
            for nu in 0..cols {
                data.push(factor.clone() * f.get(mu, nu, 0, 0).clone());
            }
        }
        GenericRMatrix::new(data, rows, cols)
    }

    /// `A^{ab}_c` as an `N(a,b,c) × N(dual(a), c, b)` matrix.
    ///
    /// Verbatim mirror of TensorKitSectors `Asymbol_from_Fsymbol`
    /// (`sectors.jl:501-511`):
    /// `reshape(√dim(a)·√dim(b)·(1/√dim(c)) ·
    ///  conj(κ_a · F(dual(a),a,b,b,unit,c)), (N(a,b,c), N(dual(a),c,b)))`.
    /// Here the *leading* two `F` axes are the singletons
    /// (`N(dual(a),a,unit) == N(unit,b,b) == 1`), so `A[κ,λ] = F[0,0,κ,λ]`.
    /// The `conj` wraps the whole `κ_a · F` product exactly as TK writes it.
    fn a_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> GenericRMatrix<Self::Scalar>
    where
        Self::Scalar: GenericBraidScalar,
    {
        let rows = self.nsymbol(a, b, c);
        let cols = self.nsymbol(self.dual(a), c, b);
        // F(dual(a), a, b, b, rightunit(a), c); `vacuum()` is the (right)unit.
        let f = self.f_symbol_generic(self.dual(a), a, b, b, self.vacuum(), c);
        let factor =
            self.sqrt_dim_scalar(a) * self.sqrt_dim_scalar(b) * self.inv_sqrt_dim_scalar(c);
        let fs = self.frobenius_schur_phase_scalar(a);
        let mut data = Vec::with_capacity(rows * cols);
        for kappa in 0..rows {
            for lambda in 0..cols {
                // conj(κ_a · F[0,0,κ,λ]), then scale by the (real) dim factor.
                let symbol = fs.clone() * f.get(0, 0, kappa, lambda).clone();
                data.push(factor.clone() * symbol.braid_conj());
            }
        }
        GenericRMatrix::new(data, rows, cols)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ProductSector<Left, Right> {
    left: Left,
    right: Right,
}

impl<Left, Right> ProductSector<Left, Right> {
    pub const fn new(left: Left, right: Right) -> Self {
        Self { left, right }
    }

    #[inline]
    pub const fn left(&self) -> &Left {
        &self.left
    }

    #[inline]
    pub const fn right(&self) -> &Right {
        &self.right
    }

    pub fn sector_id_with<C>(self) -> SectorId
    where
        C: ProductSectorCodec,
        Left: Into<SectorId>,
        Right: Into<SectorId>,
    {
        C::encode(self.left.into(), self.right.into())
    }
}

pub const fn product_sector<Left, Right>(left: Left, right: Right) -> ProductSector<Left, Right> {
    ProductSector::new(left, right)
}

pub trait ProductSectorCodec {
    fn try_encode(left: SectorId, right: SectorId) -> Option<SectorId>;

    fn encode(left: SectorId, right: SectorId) -> SectorId {
        Self::try_encode(left, right).expect("product sector id overflow")
    }

    fn decode(sector: SectorId) -> Option<(SectorId, SectorId)>;

    fn encode_checked(
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorId, ProductSectorCodecError> {
        Self::try_encode(left, right).ok_or(ProductSectorCodecError::CodecRejected)
    }

    fn decode_checked(sector: SectorId) -> Result<(SectorId, SectorId), ProductSectorCodecError> {
        Self::decode(sector).ok_or(ProductSectorCodecError::CodecRejected)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProductSectorComponent {
    Left,
    Right,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProductSectorCodecError {
    CodecRejected,
    WidthOverflow {
        left_bits: u32,
        right_bits: u32,
        available_bits: u32,
    },
    ComponentOutOfRange {
        component: ProductSectorComponent,
        sector: SectorId,
        bits: u32,
    },
    InvalidHighBits {
        sector: SectorId,
        total_bits: u32,
    },
}

impl core::fmt::Display for ProductSectorCodecError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::CodecRejected => write!(formatter, "product sector codec rejected the value"),
            Self::WidthOverflow {
                left_bits,
                right_bits,
                available_bits,
            } => write!(
                formatter,
                "product sector needs {} bits but this target provides {available_bits}",
                u64::from(*left_bits) + u64::from(*right_bits)
            ),
            Self::ComponentOutOfRange {
                component,
                sector,
                bits,
            } => write!(
                formatter,
                "{component:?} product component {sector:?} does not fit its {bits}-bit layout"
            ),
            Self::InvalidHighBits { sector, total_bits } => write!(
                formatter,
                "packed product sector {sector:?} has bits above its {total_bits}-bit layout"
            ),
        }
    }
}

impl std::error::Error for ProductSectorCodecError {}

/// Cantor-pairing product codec retained for expert compatibility.
///
/// This remains the default codec parameter of core's `ProductFusionRule` so
/// explicitly constructed expert rules keep their historical behavior.
/// Built-in user-layer product spaces use [`PackedProductCodec`] instead.
/// Cantor pairing preserves the older numeric IDs, but nested decoding is
/// magnitude-dependent and its representable component domain shrinks as
/// pairing is nested.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct TensorKitProductCodec;

impl ProductSectorCodec for TensorKitProductCodec {
    fn try_encode(left: SectorId, right: SectorId) -> Option<SectorId> {
        let left = left.id() as u128;
        let right = right.id() as u128;
        let sum = left.checked_add(right)?;
        let paired = sum
            .checked_mul(sum + 1)
            .and_then(|value| value.checked_div(2))
            .and_then(|value| value.checked_add(left))
            .and_then(|value| usize::try_from(value).ok())?;
        Some(SectorId::new(paired))
    }

    fn decode(sector: SectorId) -> Option<(SectorId, SectorId)> {
        let paired = sector.id() as u128;
        let sum = tensor_kit_product_pairing_sum(paired);
        let triangular = sum.checked_mul(sum + 1)?.checked_div(2)?;
        let left = paired.checked_sub(triangular)?;
        let right = sum.checked_sub(left)?;
        Some((
            SectorId::new(usize::try_from(left).ok()?),
            SectorId::new(usize::try_from(right).ok()?),
        ))
    }
}

fn tensor_kit_product_pairing_sum(paired: u128) -> u128 {
    let mut low = 0u128;
    let mut high = 1u128;
    while triangular_number(high) <= paired {
        high *= 2;
    }
    while low + 1 < high {
        let mid = low + (high - low) / 2;
        if triangular_number(mid) <= paired {
            low = mid;
        } else {
            high = mid;
        }
    }
    low
}

fn triangular_number(value: u128) -> u128 {
    value * (value + 1) / 2
}

/// Fixed-width leaf or recursively packed product layout.
///
/// Packed products place the left component in the low bits and the right
/// component in the high bits. A zero-bit layout can represent only ID 0;
/// a full-`usize` layout accepts every raw ID but cannot be combined with any
/// positive-width sibling. The combined width must not exceed `usize::BITS`.
///
/// [`PackedProductCodec`] independently enforces the declared raw bit width
/// before calling [`Self::validate`], so a custom validator cannot permit
/// overlapping component bits.
pub trait PackedSectorLayout {
    const BITS: u32;

    fn validate(sector: SectorId) -> Result<(), ProductSectorCodecError>;
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct Fz2SectorLayout;

impl PackedSectorLayout for Fz2SectorLayout {
    const BITS: u32 = 1;

    fn validate(sector: SectorId) -> Result<(), ProductSectorCodecError> {
        if sector.id() <= 1 {
            Ok(())
        } else {
            Err(ProductSectorCodecError::InvalidHighBits {
                sector,
                total_bits: Self::BITS,
            })
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct U1SectorLayout;

impl PackedSectorLayout for U1SectorLayout {
    const BITS: u32 = 32;

    fn validate(sector: SectorId) -> Result<(), ProductSectorCodecError> {
        if u32::try_from(sector.id()).is_ok() {
            Ok(())
        } else {
            Err(ProductSectorCodecError::InvalidHighBits {
                sector,
                total_bits: Self::BITS,
            })
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct Su2SectorLayout;

impl PackedSectorLayout for Su2SectorLayout {
    const BITS: u32 = 8;

    fn validate(sector: SectorId) -> Result<(), ProductSectorCodecError> {
        if sector.id() <= 254 {
            Ok(())
        } else {
            Err(ProductSectorCodecError::InvalidHighBits {
                sector,
                total_bits: Self::BITS,
            })
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct ProductSectorLayout<Left, Right>(PhantomData<(Left, Right)>);

impl<Left, Right> PackedSectorLayout for ProductSectorLayout<Left, Right>
where
    Left: PackedSectorLayout,
    Right: PackedSectorLayout,
{
    const BITS: u32 = Left::BITS.saturating_add(Right::BITS);

    fn validate(sector: SectorId) -> Result<(), ProductSectorCodecError> {
        let (left, right) = decode_packed_components::<Left, Right>(sector)?;
        validate_packed_component::<Left>(left, ProductSectorComponent::Left)?;
        validate_packed_component::<Right>(right, ProductSectorComponent::Right)
    }
}

/// Association-independent fixed-width product encoding.
///
/// `left` occupies bits `0..Left::BITS`; `right` occupies the next
/// `Right::BITS`. Recursive products therefore flatten to the same numeric ID
/// for `(A x B) x C` and `A x (B x C)` when the ordered leaf layouts agree.
/// The codec types, and therefore `RuleIdentity`/cache identities, remain
/// distinct across those source-level associations; numeric equality does not
/// authorize cross-provider cache reuse.
///
/// Capacity is target-dependent because the final ID is a `usize`. A layout
/// wider than `usize::BITS` is rejected with
/// [`ProductSectorCodecError::WidthOverflow`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct PackedProductCodec<Left, Right>(PhantomData<(Left, Right)>);

impl<Left, Right> ProductSectorCodec for PackedProductCodec<Left, Right>
where
    Left: PackedSectorLayout,
    Right: PackedSectorLayout,
{
    fn try_encode(left: SectorId, right: SectorId) -> Option<SectorId> {
        Self::encode_checked(left, right).ok()
    }

    fn decode(sector: SectorId) -> Option<(SectorId, SectorId)> {
        Self::decode_checked(sector).ok()
    }

    fn encode_checked(
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorId, ProductSectorCodecError> {
        validate_packed_width::<Left, Right>()?;
        validate_packed_component::<Left>(left, ProductSectorComponent::Left)?;
        validate_packed_component::<Right>(right, ProductSectorComponent::Right)?;
        let shifted_right = if Left::BITS == usize::BITS {
            0
        } else {
            right.id() << Left::BITS
        };
        Ok(SectorId::new(left.id() | shifted_right))
    }

    fn decode_checked(sector: SectorId) -> Result<(SectorId, SectorId), ProductSectorCodecError> {
        let (left, right) = decode_packed_components::<Left, Right>(sector)?;
        validate_packed_component::<Left>(left, ProductSectorComponent::Left)?;
        validate_packed_component::<Right>(right, ProductSectorComponent::Right)?;
        Ok((left, right))
    }
}

fn validate_packed_component<Layout>(
    sector: SectorId,
    component: ProductSectorComponent,
) -> Result<(), ProductSectorCodecError>
where
    Layout: PackedSectorLayout,
{
    let fits_declared_width = match Layout::BITS.cmp(&usize::BITS) {
        core::cmp::Ordering::Greater => false,
        core::cmp::Ordering::Equal => true,
        core::cmp::Ordering::Less if Layout::BITS == 0 => sector.id() == 0,
        core::cmp::Ordering::Less => sector.id() < (1usize << Layout::BITS),
    };
    if !fits_declared_width {
        return Err(ProductSectorCodecError::ComponentOutOfRange {
            component,
            sector,
            bits: Layout::BITS,
        });
    }
    Layout::validate(sector).map_err(|_| ProductSectorCodecError::ComponentOutOfRange {
        component,
        sector,
        bits: Layout::BITS,
    })
}

fn validate_packed_width<Left, Right>() -> Result<u32, ProductSectorCodecError>
where
    Left: PackedSectorLayout,
    Right: PackedSectorLayout,
{
    let total_bits =
        Left::BITS
            .checked_add(Right::BITS)
            .ok_or(ProductSectorCodecError::WidthOverflow {
                left_bits: Left::BITS,
                right_bits: Right::BITS,
                available_bits: usize::BITS,
            })?;
    if total_bits > usize::BITS {
        return Err(ProductSectorCodecError::WidthOverflow {
            left_bits: Left::BITS,
            right_bits: Right::BITS,
            available_bits: usize::BITS,
        });
    }
    Ok(total_bits)
}

fn decode_packed_components<Left, Right>(
    sector: SectorId,
) -> Result<(SectorId, SectorId), ProductSectorCodecError>
where
    Left: PackedSectorLayout,
    Right: PackedSectorLayout,
{
    let total_bits = validate_packed_width::<Left, Right>()?;
    if total_bits < usize::BITS && sector.id() >> total_bits != 0 {
        return Err(ProductSectorCodecError::InvalidHighBits { sector, total_bits });
    }
    let left_mask = if Left::BITS == usize::BITS {
        usize::MAX
    } else {
        (1usize << Left::BITS) - 1
    };
    let right = if Left::BITS == usize::BITS {
        0
    } else {
        sector.id() >> Left::BITS
    };
    Ok((SectorId::new(sector.id() & left_mask), SectorId::new(right)))
}
