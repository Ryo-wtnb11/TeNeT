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

#[derive(Clone, Debug)]
pub struct RuleIdentity(RuleIdentityNode);

#[derive(Clone, Debug)]
enum RuleIdentityNode {
    Type(std::any::TypeId),
    Unique(std::any::TypeId, u64),
    Content {
        rule_type: std::any::TypeId,
        prehash: u64,
        bytes: Arc<[u8]>,
    },
    Product(Arc<ProductRuleIdentity>),
}

#[derive(Debug, Eq, Hash, PartialEq)]
struct ProductRuleIdentity {
        codec: std::any::TypeId,
        left: RuleIdentity,
        right: RuleIdentity,
}

impl PartialEq for RuleIdentity {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for RuleIdentity {}

impl std::hash::Hash for RuleIdentity {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl PartialEq for RuleIdentityNode {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Type(left), Self::Type(right)) => left == right,
            (Self::Unique(left_type, left), Self::Unique(right_type, right)) => {
                left_type == right_type && left == right
            }
            (
                Self::Content { rule_type: left_type, prehash: left_hash, bytes: left },
                Self::Content { rule_type: right_type, prehash: right_hash, bytes: right },
            ) => {
                left_type == right_type
                    && left_hash == right_hash
                    && (Arc::ptr_eq(left, right) || left.as_ref() == right.as_ref())
            }
            (Self::Product(left), Self::Product(right)) => left == right,
            _ => false,
        }
    }
}

impl Eq for RuleIdentityNode {}

impl std::hash::Hash for RuleIdentityNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Type(rule_type) => rule_type.hash(state),
            Self::Unique(rule_type, instance) => {
                rule_type.hash(state);
                instance.hash(state);
            }
            Self::Content { rule_type, prehash, .. } => {
                rule_type.hash(state);
                prehash.hash(state);
            }
            Self::Product(identity) => identity.hash(state),
        }
    }
}

impl RuleIdentity {
    pub fn of_type<R: 'static + ?Sized>() -> Self {
        Self(RuleIdentityNode::Type(std::any::TypeId::of::<R>()))
    }

    pub fn new_unique<R: 'static>() -> Self {
        static NEXT_INSTANCE: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(1);
        let instance = NEXT_INSTANCE
            .fetch_update(
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
                |current| current.checked_add(1),
            )
            .expect("fusion-rule identity space exhausted");
        Self(RuleIdentityNode::Unique(
            std::any::TypeId::of::<R>(),
            instance,
        ))
    }

    pub fn from_canonical_bytes<R: 'static>(prehash: u64, bytes: Arc<[u8]>) -> Self {
        Self(RuleIdentityNode::Content {
            rule_type: std::any::TypeId::of::<R>(),
            prehash,
            bytes,
        })
    }

    fn product<Codec: 'static>(left: Self, right: Self) -> Self {
        Self(RuleIdentityNode::Product(Arc::new(ProductRuleIdentity {
            codec: std::any::TypeId::of::<Codec>(),
            left,
            right,
        })))
    }
}

pub trait FusionRule: 'static {
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

pub trait MultiplicityFreeFusionRule: FusionRule {}

pub trait MultiplicityFreeFusionSymbols: MultiplicityFreeFusionRule {
    // Send + Sync because cached recoupling coefficients are shared across
    // tree-transform replay workers (TensorKit sectorscalartype parity: the
    // concrete scalar is a plain number type).
    type Scalar: Clone + Send + Sync;

    fn scalar_one(&self) -> Self::Scalar;

    fn scalar_conj(&self, value: Self::Scalar) -> Self::Scalar;

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

pub trait MultiplicityFreePivotalSymbols: MultiplicityFreeFusionSymbols {
    fn bendright_scalar(
        &self,
        left_coupled: SectorId,
        bent_sector: SectorId,
        coupled: SectorId,
        bent_leg_is_dual: bool,
    ) -> Self::Scalar;

    fn foldright_scalar(
        &self,
        source: &FusionTreeBlockKey,
        destination: &FusionTreeBlockKey,
    ) -> Self::Scalar;
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

impl<Scalar> GenericFArray<Scalar> {
    pub fn new(data: Vec<Scalar>, shape: (usize, usize, usize, usize)) -> Self {
        let (n_mu, n_nu, n_kappa, n_lambda) = shape;
        debug_assert_eq!(
            data.len(),
            n_mu * n_nu * n_kappa * n_lambda,
            "GenericFArray data length must match shape product"
        );
        Self { data, shape }
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
        debug_assert_eq!(
            data.len(),
            rows * cols,
            "GenericRMatrix data length must match rows * cols"
        );
        Self { data, rows, cols }
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

    fn r_symbol_generic(&self, a: SectorId, b: SectorId, c: SectorId)
        -> GenericRMatrix<Self::Scalar>;
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
        let factor = self.sqrt_dim_scalar(a)
            * self.sqrt_dim_scalar(b)
            * self.inv_sqrt_dim_scalar(c);
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
        let factor = self.sqrt_dim_scalar(a)
            * self.sqrt_dim_scalar(b)
            * self.inv_sqrt_dim_scalar(c);
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
}

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

#[derive(Clone, Debug)]
pub struct ProductFusionRule<LeftRule, RightRule, Codec = TensorKitProductCodec> {
    left: LeftRule,
    right: RightRule,
    _codec: PhantomData<Codec>,
    identity: OnceLock<RuleIdentity>,
}

impl<LeftRule, RightRule, Codec> ProductFusionRule<LeftRule, RightRule, Codec> {
    pub const fn new(left: LeftRule, right: RightRule) -> Self {
        Self {
            left,
            right,
            _codec: PhantomData,
            identity: OnceLock::new(),
        }
    }

    #[inline]
    pub const fn left_rule(&self) -> &LeftRule {
        &self.left
    }

    #[inline]
    pub const fn right_rule(&self) -> &RightRule {
        &self.right
    }

    pub fn encode_sector(&self, left: SectorId, right: SectorId) -> SectorId
    where
        Codec: ProductSectorCodec,
    {
        Codec::encode(left, right)
    }

    pub fn decode_sector(&self, sector: SectorId) -> Option<(SectorId, SectorId)>
    where
        Codec: ProductSectorCodec,
    {
        Codec::decode(sector)
    }

    fn decode_sector_or_panic(&self, sector: SectorId) -> (SectorId, SectorId)
    where
        Codec: ProductSectorCodec,
    {
        self.decode_sector(sector)
            .expect("product fusion rule received an invalid product sector")
    }
}

pub const fn product_fusion_rule<LeftRule, RightRule>(
    left: LeftRule,
    right: RightRule,
) -> ProductFusionRule<LeftRule, RightRule> {
    ProductFusionRule::new(left, right)
}

pub const fn product_fusion_rule_with_codec<LeftRule, RightRule, Codec>(
    left: LeftRule,
    right: RightRule,
) -> ProductFusionRule<LeftRule, RightRule, Codec> {
    ProductFusionRule::new(left, right)
}

pub trait ProductFusionRuleExt: FusionRule + Sized {
    fn product<RightRule>(self, right: RightRule) -> ProductFusionRule<Self, RightRule>
    where
        RightRule: FusionRule,
    {
        ProductFusionRule::new(self, right)
    }
}

impl<Rule> ProductFusionRuleExt for Rule where Rule: FusionRule + Sized {}

impl<LeftRule, RightRule, Codec> Default for ProductFusionRule<LeftRule, RightRule, Codec>
where
    LeftRule: Default,
    RightRule: Default,
{
    fn default() -> Self {
        Self::new(LeftRule::default(), RightRule::default())
    }
}

impl<LeftRule, RightRule, Codec> FusionRule for ProductFusionRule<LeftRule, RightRule, Codec>
where
    LeftRule: FusionRule,
    RightRule: FusionRule,
    Codec: ProductSectorCodec + 'static,
{
    fn rule_identity(&self) -> RuleIdentity {
        self.identity
            .get_or_init(|| {
                RuleIdentity::product::<Codec>(
                    self.left.rule_identity(),
                    self.right.rule_identity(),
                )
            })
            .clone()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        self.left
            .fusion_style()
            .combined_with(self.right.fusion_style())
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        self.left
            .braiding_style()
            .combined_with(self.right.braiding_style())
    }

    fn vacuum(&self) -> SectorId {
        self.encode_sector(self.left.vacuum(), self.right.vacuum())
    }

    fn supports_unitary_braid_dagger(&self) -> bool {
        self.left.supports_unitary_braid_dagger() && self.right.supports_unitary_braid_dagger()
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        let (left, right) = self.decode_sector_or_panic(sector);
        self.encode_sector(self.left.dual(left), self.right.dual(right))
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        let (left_left, left_right) = self.decode_sector_or_panic(left);
        let (right_left, right_right) = self.decode_sector_or_panic(right);
        let left_channels = self.left.fusion_channels(left_left, right_left);
        let right_channels = self.right.fusion_channels(left_right, right_right);
        // Cartesian product of the two sub-rules' channels, matching TensorKit's
        // `⊗(p1,p2) = SectorSet(product(map(⊗, ...)))`. No dedup: each sub-rule
        // is multiplicity-free (distinct channels) and `encode_sector` is the
        // Cantor pairing (a bijection), so distinct (left,right) pairs always
        // encode to distinct ids — the old `channels.contains()` guard was
        // provably dead and made this O(k²) instead of O(k) in k = |L|·|R|.
        let mut channels = SectorVec::with_capacity(left_channels.len() * right_channels.len());
        for right_channel in right_channels {
            for &left_channel in &left_channels {
                channels.push(self.encode_sector(left_channel, right_channel));
            }
        }
        channels
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        let (left_left, left_right) = self.decode_sector_or_panic(left);
        let (right_left, right_right) = self.decode_sector_or_panic(right);
        let (coupled_left, coupled_right) = self.decode_sector_or_panic(coupled);
        self.left.nsymbol(left_left, right_left, coupled_left)
            * self.right.nsymbol(left_right, right_right, coupled_right)
    }
}

impl<LeftRule, RightRule, Codec> MultiplicityFreeFusionRule
    for ProductFusionRule<LeftRule, RightRule, Codec>
where
    LeftRule: MultiplicityFreeFusionRule,
    RightRule: MultiplicityFreeFusionRule,
    Codec: ProductSectorCodec + 'static,
{
}

impl<LeftRule, RightRule, Codec> MultiplicityFreeFusionSymbols
    for ProductFusionRule<LeftRule, RightRule, Codec>
where
    LeftRule: MultiplicityFreeFusionSymbols<Scalar = f64>,
    RightRule: MultiplicityFreeFusionSymbols<Scalar = f64>,
    Codec: ProductSectorCodec + 'static,
{
    type Scalar = f64;

    fn scalar_one(&self) -> Self::Scalar {
        1.0
    }

    fn scalar_conj(&self, value: Self::Scalar) -> Self::Scalar {
        value
    }

    fn f_symbol_scalar(
        &self,
        left: SectorId,
        middle: SectorId,
        right: SectorId,
        coupled: SectorId,
        left_coupled: SectorId,
        right_coupled: SectorId,
    ) -> Self::Scalar {
        let (left_l, left_r) = self.decode_sector_or_panic(left);
        let (middle_l, middle_r) = self.decode_sector_or_panic(middle);
        let (right_l, right_r) = self.decode_sector_or_panic(right);
        let (coupled_l, coupled_r) = self.decode_sector_or_panic(coupled);
        let (left_coupled_l, left_coupled_r) = self.decode_sector_or_panic(left_coupled);
        let (right_coupled_l, right_coupled_r) = self.decode_sector_or_panic(right_coupled);
        self.left.f_symbol_scalar(
            left_l,
            middle_l,
            right_l,
            coupled_l,
            left_coupled_l,
            right_coupled_l,
        ) * self.right.f_symbol_scalar(
            left_r,
            middle_r,
            right_r,
            coupled_r,
            left_coupled_r,
            right_coupled_r,
        )
    }

    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar {
        let (left_l, left_r) = self.decode_sector_or_panic(left);
        let (right_l, right_r) = self.decode_sector_or_panic(right);
        let (coupled_l, coupled_r) = self.decode_sector_or_panic(coupled);
        self.left.r_symbol_scalar(left_l, right_l, coupled_l)
            * self.right.r_symbol_scalar(left_r, right_r, coupled_r)
    }
}

impl<LeftRule, RightRule, Codec> MultiplicityFreeRigidSymbols
    for ProductFusionRule<LeftRule, RightRule, Codec>
where
    LeftRule: MultiplicityFreeRigidSymbols<Scalar = f64>,
    RightRule: MultiplicityFreeRigidSymbols<Scalar = f64>,
    // Sync via the trait's supertrait; the codec is a PhantomData marker.
    Codec: ProductSectorCodec + Sync + 'static,
{
    fn dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        let (left, right) = self.decode_sector_or_panic(sector);
        self.left.dim_scalar(left) * self.right.dim_scalar(right)
    }

    fn inv_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        let (left, right) = self.decode_sector_or_panic(sector);
        self.left.inv_dim_scalar(left) * self.right.inv_dim_scalar(right)
    }

    fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        let (left, right) = self.decode_sector_or_panic(sector);
        self.left.sqrt_dim_scalar(left) * self.right.sqrt_dim_scalar(right)
    }

    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        let (left, right) = self.decode_sector_or_panic(sector);
        self.left.inv_sqrt_dim_scalar(left) * self.right.inv_sqrt_dim_scalar(right)
    }

    fn twist_scalar(&self, sector: SectorId) -> Self::Scalar {
        let (left, right) = self.decode_sector_or_panic(sector);
        self.left.twist_scalar(left) * self.right.twist_scalar(right)
    }

    fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> Self::Scalar {
        let (left, right) = self.decode_sector_or_panic(sector);
        self.left.frobenius_schur_phase_scalar(left)
            * self.right.frobenius_schur_phase_scalar(right)
    }

    fn a_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar {
        let (left_l, left_r) = self.decode_sector_or_panic(left);
        let (right_l, right_r) = self.decode_sector_or_panic(right);
        let (coupled_l, coupled_r) = self.decode_sector_or_panic(coupled);
        self.left.a_symbol_scalar(left_l, right_l, coupled_l)
            * self.right.a_symbol_scalar(left_r, right_r, coupled_r)
    }

    fn b_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar {
        let (left_l, left_r) = self.decode_sector_or_panic(left);
        let (right_l, right_r) = self.decode_sector_or_panic(right);
        let (coupled_l, coupled_r) = self.decode_sector_or_panic(coupled);
        self.left.b_symbol_scalar(left_l, right_l, coupled_l)
            * self.right.b_symbol_scalar(left_r, right_r, coupled_r)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Z2Irrep {
    parity: u8,
}

impl Z2Irrep {
    pub const EVEN: Self = Self { parity: 0 };
    pub const ODD: Self = Self { parity: 1 };

    pub const fn new(parity: u8) -> Self {
        Self { parity: parity & 1 }
    }

    #[inline]
    pub const fn parity(self) -> u8 {
        self.parity
    }

    #[inline]
    pub const fn sector_id(self) -> SectorId {
        SectorId::new(self.parity as usize)
    }

    pub const fn from_sector_id(sector: SectorId) -> Option<Self> {
        match sector.id() {
            0 => Some(Self::EVEN),
            1 => Some(Self::ODD),
            _ => None,
        }
    }
}

impl From<Z2Irrep> for SectorId {
    fn from(value: Z2Irrep) -> Self {
        value.sector_id()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct Z2FusionRule;

impl FusionRule for Z2FusionRule {
    fn rule_identity(&self) -> RuleIdentity { RuleIdentity::of_type::<Self>() }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        Z2Irrep::EVEN.into()
    }

    fn supports_unitary_braid_dagger(&self) -> bool {
        true
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        let left = Z2Irrep::from_sector_id(left).expect("Z2 fusion received an invalid sector");
        let right = Z2Irrep::from_sector_id(right).expect("Z2 fusion received an invalid sector");
        core::iter::once(Z2Irrep::new(left.parity() ^ right.parity()).into()).collect()
    }
}

impl MultiplicityFreeFusionRule for Z2FusionRule {}

impl MultiplicityFreeFusionSymbols for Z2FusionRule {
    type Scalar = f64;

    fn scalar_one(&self) -> Self::Scalar {
        1.0
    }

    fn scalar_conj(&self, value: Self::Scalar) -> Self::Scalar {
        value
    }

    fn f_symbol_scalar(
        &self,
        _left: SectorId,
        _middle: SectorId,
        _right: SectorId,
        _coupled: SectorId,
        _left_coupled: SectorId,
        _right_coupled: SectorId,
    ) -> Self::Scalar {
        1.0
    }

    fn r_symbol_scalar(
        &self,
        _left: SectorId,
        _right: SectorId,
        _coupled: SectorId,
    ) -> Self::Scalar {
        1.0
    }
}

impl MultiplicityFreePivotalSymbols for Z2FusionRule {
    fn bendright_scalar(
        &self,
        _left_coupled: SectorId,
        _bent_sector: SectorId,
        _coupled: SectorId,
        _bent_leg_is_dual: bool,
    ) -> Self::Scalar {
        1.0
    }

    fn foldright_scalar(
        &self,
        _source: &FusionTreeBlockKey,
        _destination: &FusionTreeBlockKey,
    ) -> Self::Scalar {
        1.0
    }
}

impl MultiplicityFreeRigidSymbols for Z2FusionRule {
    fn dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn inv_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn twist_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct FermionParityFusionRule;

impl FusionRule for FermionParityFusionRule {
    fn rule_identity(&self) -> RuleIdentity { RuleIdentity::of_type::<Self>() }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Fermionic
    }

    fn vacuum(&self) -> SectorId {
        Z2Irrep::EVEN.into()
    }

    fn supports_unitary_braid_dagger(&self) -> bool {
        true
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        Z2FusionRule.fusion_channels(left, right)
    }
}

impl MultiplicityFreeFusionRule for FermionParityFusionRule {}

impl MultiplicityFreeFusionSymbols for FermionParityFusionRule {
    type Scalar = f64;

    fn scalar_one(&self) -> Self::Scalar {
        1.0
    }

    fn scalar_conj(&self, value: Self::Scalar) -> Self::Scalar {
        value
    }

    fn f_symbol_scalar(
        &self,
        _left: SectorId,
        _middle: SectorId,
        _right: SectorId,
        _coupled: SectorId,
        _left_coupled: SectorId,
        _right_coupled: SectorId,
    ) -> Self::Scalar {
        1.0
    }

    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, _coupled: SectorId) -> Self::Scalar {
        if left == Z2Irrep::ODD.into() && right == Z2Irrep::ODD.into() {
            -1.0
        } else {
            1.0
        }
    }
}

impl MultiplicityFreePivotalSymbols for FermionParityFusionRule {
    fn bendright_scalar(
        &self,
        _left_coupled: SectorId,
        _bent_sector: SectorId,
        _coupled: SectorId,
        _bent_leg_is_dual: bool,
    ) -> Self::Scalar {
        1.0
    }

    fn foldright_scalar(
        &self,
        _source: &FusionTreeBlockKey,
        _destination: &FusionTreeBlockKey,
    ) -> Self::Scalar {
        1.0
    }
}

impl MultiplicityFreeRigidSymbols for FermionParityFusionRule {
    fn dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn inv_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn twist_scalar(&self, sector: SectorId) -> Self::Scalar {
        if sector == Z2Irrep::ODD.into() {
            -1.0
        } else {
            1.0
        }
    }

    fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct U1Irrep {
    charge: i32,
}

impl U1Irrep {
    pub const fn new(charge: i32) -> Self {
        Self { charge }
    }

    #[inline]
    pub const fn charge(self) -> i32 {
        self.charge
    }

    pub const fn sector_id(self) -> SectorId {
        let charge = self.charge as i64;
        if charge >= 0 {
            SectorId::new((charge as usize) * 2)
        } else {
            SectorId::new(((-charge) as usize) * 2 - 1)
        }
    }

    pub fn from_sector_id(sector: SectorId) -> Option<Self> {
        let id = sector.id();
        if id > u32::MAX as usize {
            return None;
        }
        let charge = if id % 2 == 0 {
            i64::try_from(id / 2).ok()?
        } else {
            -i64::try_from((id + 1) / 2).ok()?
        };
        i32::try_from(charge).ok().map(Self::new)
    }
}

impl From<U1Irrep> for SectorId {
    fn from(value: U1Irrep) -> Self {
        value.sector_id()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct U1FusionRule;

impl FusionRule for U1FusionRule {
    fn rule_identity(&self) -> RuleIdentity { RuleIdentity::of_type::<Self>() }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        U1Irrep::new(0).into()
    }

    fn supports_unitary_braid_dagger(&self) -> bool {
        true
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        let sector = U1Irrep::from_sector_id(sector).expect("U(1) dual received an invalid sector");
        U1Irrep::new(
            sector
                .charge()
                .checked_neg()
                .expect("U(1) dual charge overflow"),
        )
        .into()
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        let left = U1Irrep::from_sector_id(left).expect("U(1) fusion received an invalid sector");
        let right = U1Irrep::from_sector_id(right).expect("U(1) fusion received an invalid sector");
        core::iter::once(
            U1Irrep::new(
                left.charge()
                    .checked_add(right.charge())
                    .expect("U(1) fusion charge overflow"),
            )
            .into(),
        )
        .collect()
    }
}

impl MultiplicityFreeFusionRule for U1FusionRule {}

impl MultiplicityFreeFusionSymbols for U1FusionRule {
    type Scalar = f64;

    fn scalar_one(&self) -> Self::Scalar {
        1.0
    }

    fn scalar_conj(&self, value: Self::Scalar) -> Self::Scalar {
        value
    }

    fn f_symbol_scalar(
        &self,
        _left: SectorId,
        _middle: SectorId,
        _right: SectorId,
        _coupled: SectorId,
        _left_coupled: SectorId,
        _right_coupled: SectorId,
    ) -> Self::Scalar {
        1.0
    }

    fn r_symbol_scalar(
        &self,
        _left: SectorId,
        _right: SectorId,
        _coupled: SectorId,
    ) -> Self::Scalar {
        1.0
    }
}

impl MultiplicityFreeRigidSymbols for U1FusionRule {
    fn dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn inv_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn twist_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SU2Irrep {
    twice_spin: usize,
}

impl SU2Irrep {
    pub const fn from_twice_spin(twice_spin: usize) -> Self {
        Self { twice_spin }
    }

    #[inline]
    pub const fn twice_spin(self) -> usize {
        self.twice_spin
    }

    #[inline]
    pub const fn sector_id(self) -> SectorId {
        SectorId::new(self.twice_spin)
    }

    pub const fn from_sector_id(sector: SectorId) -> Self {
        Self::from_twice_spin(sector.id())
    }
}

impl From<SU2Irrep> for SectorId {
    fn from(value: SU2Irrep) -> Self {
        value.sector_id()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct SU2FusionRule;

impl FusionRule for SU2FusionRule {
    fn rule_identity(&self) -> RuleIdentity { RuleIdentity::of_type::<Self>() }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Simple
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        SU2Irrep::from_twice_spin(0).into()
    }

    fn supports_unitary_braid_dagger(&self) -> bool {
        true
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        let left = SU2Irrep::from_sector_id(left).twice_spin();
        let right = SU2Irrep::from_sector_id(right).twice_spin();
        let min = left.abs_diff(right);
        let max = left + right;
        (min..=max)
            .step_by(2)
            .map(|twice_spin| SU2Irrep::from_twice_spin(twice_spin).into())
            .collect()
    }
}

impl MultiplicityFreeFusionRule for SU2FusionRule {}

impl MultiplicityFreeFusionSymbols for SU2FusionRule {
    type Scalar = f64;

    fn scalar_one(&self) -> Self::Scalar {
        1.0
    }

    fn scalar_conj(&self, value: Self::Scalar) -> Self::Scalar {
        value
    }

    fn f_symbol_scalar(
        &self,
        left: SectorId,
        middle: SectorId,
        right: SectorId,
        coupled: SectorId,
        left_coupled: SectorId,
        right_coupled: SectorId,
    ) -> Self::Scalar {
        let j1 = SU2Irrep::from_sector_id(left).twice_spin();
        let j2 = SU2Irrep::from_sector_id(middle).twice_spin();
        let j3 = SU2Irrep::from_sector_id(right).twice_spin();
        let j4 = SU2Irrep::from_sector_id(coupled).twice_spin();
        let j5 = SU2Irrep::from_sector_id(left_coupled).twice_spin();
        let j6 = SU2Irrep::from_sector_id(right_coupled).twice_spin();
        su2_f_symbol_from_doubled_spins(j1, j2, j3, j4, j5, j6)
    }

    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar {
        if self.nsymbol(left, right, coupled) == 0 {
            return 0.0;
        }
        let left = SU2Irrep::from_sector_id(left).twice_spin();
        let right = SU2Irrep::from_sector_id(right).twice_spin();
        let coupled = SU2Irrep::from_sector_id(coupled).twice_spin();
        let exponent = (left + right - coupled) / 2;
        if exponent % 2 == 0 {
            1.0
        } else {
            -1.0
        }
    }
}

impl MultiplicityFreeRigidSymbols for SU2FusionRule {
    fn dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        (SU2Irrep::from_sector_id(sector).twice_spin() + 1) as f64
    }

    fn inv_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        1.0 / self.dim_scalar(sector)
    }

    fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        self.dim_scalar(sector).sqrt()
    }

    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        1.0 / self.sqrt_dim_scalar(sector)
    }

    fn twist_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> Self::Scalar {
        if SU2Irrep::from_sector_id(sector).twice_spin() % 2 == 0 {
            1.0
        } else {
            -1.0
        }
    }
}

fn su2_f_symbol_from_doubled_spins(
    j1: usize,
    j2: usize,
    j3: usize,
    j4: usize,
    j5: usize,
    j6: usize,
) -> f64 {
    if [j1, j2, j3, j4, j5, j6].iter().all(|&j| j == 0) {
        return 1.0;
    }
    let phase_exponent = (j1 + j2 + j3 + j4) / 2;
    let phase = if phase_exponent % 2 == 0 { 1.0 } else { -1.0 };
    let dimension_factor = (((j5 + 1) * (j6 + 1)) as f64).sqrt();
    phase * dimension_factor * wigner_6j_doubled(j1, j2, j5, j3, j4, j6)
}

/// SU(2) 6j symbol (arguments as doubled spins), memoized in a process-global
/// cache keyed by the six doubled spins — the analogue of TensorKit's
/// `WignerSymbols.Wigner6j` LRU. Each distinct symbol's exact summation is
/// evaluated once; every later occurrence (across braids, permutes, and
/// contractions) is a hash lookup. The cached value is bit-identical to the
/// direct computation, so this changes cold compile cost only.
fn wigner_6j_doubled(j1: usize, j2: usize, j3: usize, j4: usize, j5: usize, j6: usize) -> f64 {
    static CACHE: std::sync::OnceLock<std::sync::RwLock<FxHashMap<[usize; 6], f64>>> =
        std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::RwLock::new(FxHashMap::default()));
    let key = [j1, j2, j3, j4, j5, j6];
    if let Ok(read) = cache.read() {
        if let Some(&value) = read.get(&key) {
            return value;
        }
    }
    let value = wigner_6j_doubled_uncached(j1, j2, j3, j4, j5, j6);
    if let Ok(mut write) = cache.write() {
        write.insert(key, value);
    }
    value
}

fn wigner_6j_doubled_uncached(
    j1: usize,
    j2: usize,
    j3: usize,
    j4: usize,
    j5: usize,
    j6: usize,
) -> f64 {
    let Some(delta_ln) = su2_delta_ln(j1, j2, j3)
        .and_then(|value| su2_delta_ln(j1, j5, j6).map(|next| value + next))
        .and_then(|value| su2_delta_ln(j4, j2, j6).map(|next| value + next))
        .and_then(|value| su2_delta_ln(j4, j5, j3).map(|next| value + next))
    else {
        return 0.0;
    };

    let x = [j1 + j2 + j3, j1 + j5 + j6, j4 + j2 + j6, j4 + j5 + j3];
    let y = [j1 + j2 + j4 + j5, j1 + j3 + j4 + j6, j2 + j3 + j5 + j6];
    let mut z_min = x.into_iter().max().unwrap_or(0);
    let z_max = y.into_iter().min().unwrap_or(0);
    if z_min > z_max {
        return 0.0;
    }
    if z_min % 2 != 0 {
        z_min += 1;
    }

    let mut sum = 0.0;
    let mut z_doubled = z_min;
    while z_doubled <= z_max {
        let z = z_doubled / 2;
        let mut term_ln = ln_factorial(z + 1);
        for value in x {
            term_ln -= ln_factorial((z_doubled - value) / 2);
        }
        for value in y {
            term_ln -= ln_factorial((value - z_doubled) / 2);
        }
        let sign = if z % 2 == 0 { 1.0 } else { -1.0 };
        sum += sign * term_ln.exp();
        z_doubled += 2;
    }

    delta_ln.exp() * sum
}

fn su2_delta_ln(j1: usize, j2: usize, j3: usize) -> Option<f64> {
    if (j1 + j2 + j3) % 2 != 0 {
        return None;
    }
    if j1 + j2 < j3 || j1 + j3 < j2 || j2 + j3 < j1 {
        return None;
    }
    Some(
        0.5 * (ln_factorial((j1 + j2 - j3) / 2)
            + ln_factorial((j1 + j3 - j2) / 2)
            + ln_factorial((j2 + j3 - j1) / 2)
            - ln_factorial((j1 + j2 + j3) / 2 + 1)),
    )
}

/// `ln(n!)`, memoized in a process-global lazily-extended table.
///
/// `ln(n!) = ln((n-1)!) + ln(n)` is monotone, so the table is filled once and
/// every later call is an O(1) lookup. Recoupling-coefficient evaluation
/// (6j symbols) calls this ~7x per summation term, so cold structure compile
/// dominated by the previous naive `(1..=n)` recomputation. The values are
/// identical; this only removes the recomputation.
fn ln_factorial(n: usize) -> f64 {
    static TABLE: std::sync::OnceLock<std::sync::RwLock<Vec<f64>>> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| std::sync::RwLock::new(vec![0.0]));
    if let Ok(read) = table.read() {
        if let Some(&value) = read.get(n) {
            return value;
        }
    }
    let mut write = table.write().expect("ln_factorial table poisoned");
    while write.len() <= n {
        let previous = *write.last().expect("table seeded with ln(0!) = 0");
        let next = write.len();
        write.push(previous + (next as f64).ln());
    }
    write[n]
}

// FibonacciAnyon: the simplest genuinely non-abelian anyon model (Simple
// fusion + Anyonic braiding + complex F/R symbols) — SectorId 0 = vacuum
// `𝟙`, SectorId 1 = `τ`, with `τ⊗τ = 𝟙 ⊕ τ`. All numeric F/R/dim/twist
// values below are copied verbatim from TensorKitSectors.jl's
// `FibonacciAnyon` (`~/.julia/packages/TensorKitSectors/tugbK/src/anyons.jl`,
// lines 82-146) — never "simplify" a sign or phase here without rereading
// that source (project convention: don't derive anyon conventions from
// "should be").
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct FibonacciFusionRule;

impl FibonacciFusionRule {
    /// `false` for the vacuum (`𝟙`, SectorId 0), `true` for `τ` (SectorId 1).
    fn is_tau(sector: SectorId) -> bool {
        sector.id() != 0
    }
}

/// `dim(FibonacciAnyon)` (anyons.jl:82-83): `𝟙 -> 1`, `τ -> φ = (1+√5)/2`
/// (`Float64(MathConstants.golden)`).
fn fibonacci_quantum_dim(sector: SectorId) -> f64 {
    if FibonacciFusionRule::is_tau(sector) {
        (1.0 + 5.0_f64.sqrt()) / 2.0
    } else {
        1.0
    }
}

impl FusionRule for FibonacciFusionRule {
    fn rule_identity(&self) -> RuleIdentity { RuleIdentity::of_type::<Self>() }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Simple
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Anyonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    // `dual(s) = s` (anyons.jl:80: `dual(s::FibonacciAnyon) = s`) is exactly
    // the `FusionRule::dual` default (identity) — no override needed.

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        match (Self::is_tau(left), Self::is_tau(right)) {
            (false, _) => smallvec![right],
            (true, false) => smallvec![left],
            // τ⊗τ = 𝟙 ⊕ τ, vacuum-first to match TensorKitSectors'
            // `FibonacciAnyonProdIterator` iteration order (anyons.jl:96-109).
            (true, true) => smallvec![SectorId::new(0), SectorId::new(1)],
        }
    }
}

impl MultiplicityFreeFusionRule for FibonacciFusionRule {}

impl MultiplicityFreeFusionSymbols for FibonacciFusionRule {
    type Scalar = Complex64;

    fn scalar_one(&self) -> Self::Scalar {
        Complex64::new(1.0, 0.0)
    }

    fn scalar_conj(&self, value: Self::Scalar) -> Self::Scalar {
        value.conj()
    }

    // Verbatim port of `Fsymbol` (anyons.jl:115-137): four `Nsymbol` gates,
    // then the single non-trivial 2x2 block `F^{τττ}_τ` (entries ±1/φ,
    // ±1/√φ); every other allowed configuration is 1.
    fn f_symbol_scalar(
        &self,
        left: SectorId,
        middle: SectorId,
        right: SectorId,
        coupled: SectorId,
        left_coupled: SectorId,
        right_coupled: SectorId,
    ) -> Self::Scalar {
        if self.nsymbol(left, middle, left_coupled) == 0
            || self.nsymbol(left_coupled, right, coupled) == 0
            || self.nsymbol(middle, right, right_coupled) == 0
            || self.nsymbol(left, right_coupled, coupled) == 0
        {
            return Complex64::new(0.0, 0.0);
        }
        if Self::is_tau(left) && Self::is_tau(middle) && Self::is_tau(right) && Self::is_tau(coupled)
        {
            let phi = fibonacci_quantum_dim(SectorId::new(1));
            if !Self::is_tau(left_coupled) && !Self::is_tau(right_coupled) {
                Complex64::new(1.0 / phi, 0.0)
            } else if Self::is_tau(left_coupled) && Self::is_tau(right_coupled) {
                Complex64::new(-1.0 / phi, 0.0)
            } else {
                Complex64::new(1.0 / phi.sqrt(), 0.0)
            }
        } else {
            Complex64::new(1.0, 0.0)
        }
    }

    // Verbatim port of `Rsymbol` (anyons.jl:139-146): trivial braiding with
    // the vacuum, and the two complex phases `cispi(4/5)` / `cispi(-3/5)`
    // for `R^{ττ}_𝟙` / `R^{ττ}_τ`.
    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar {
        if self.nsymbol(left, right, coupled) == 0 {
            return Complex64::new(0.0, 0.0);
        }
        if !Self::is_tau(left) || !Self::is_tau(right) {
            Complex64::new(1.0, 0.0)
        } else if !Self::is_tau(coupled) {
            Complex64::from_polar(1.0, std::f64::consts::PI * 4.0 / 5.0)
        } else {
            Complex64::from_polar(1.0, std::f64::consts::PI * -3.0 / 5.0)
        }
    }
}

impl MultiplicityFreePivotalSymbols for FibonacciFusionRule {
    // Dead code for this provider: every `unique_*` caller of
    // `bendright_scalar`/`foldright_scalar` gates on
    // `fusion_style() == FusionStyleKind::Unique` and errors out first
    // (verified by reading every call site), and `FibonacciFusionRule` is
    // `Simple`. The Simple-fusion bend path
    // (`multiplicity_free_bendright_tree_pair`) instead derives its
    // coefficient from `b_symbol_scalar`/`sqrt_dim_scalar`, which Fibonacci
    // gets for free from `MultiplicityFreeRigidSymbols` below. Implemented
    // here only for interface parity with the sibling providers
    // (Z2/Fermion/AsymmetricAnyonic), which all also return the scalar unit.
    fn bendright_scalar(
        &self,
        _left_coupled: SectorId,
        _bent_sector: SectorId,
        _coupled: SectorId,
        _bent_leg_is_dual: bool,
    ) -> Self::Scalar {
        self.scalar_one()
    }

    fn foldright_scalar(
        &self,
        _source: &FusionTreeBlockKey,
        _destination: &FusionTreeBlockKey,
    ) -> Self::Scalar {
        self.scalar_one()
    }
}

impl MultiplicityFreeRigidSymbols for FibonacciFusionRule {
    fn dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        Complex64::new(fibonacci_quantum_dim(sector), 0.0)
    }

    fn inv_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        Complex64::new(1.0 / fibonacci_quantum_dim(sector), 0.0)
    }

    fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        Complex64::new(fibonacci_quantum_dim(sector).sqrt(), 0.0)
    }

    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        Complex64::new(1.0 / fibonacci_quantum_dim(sector).sqrt(), 0.0)
    }

    // TensorKitSectors has no `FibonacciAnyon`-specific `twist` override, so
    // it falls back to the generic `twist_from_Rsymbol` (sectors.jl:646-647):
    // `twist(a) = Σ_{b ∈ a⊗a} dim(b)/dim(a) * Rsymbol(a,a,b)`. Verified
    // numerically against that formula (not guessed):
    //   twist(𝟙) = 1
    //   twist(τ) = (1/φ)·cispi(4/5) + (φ/φ)·cispi(-3/5) = cispi(-4/5)
    fn twist_scalar(&self, sector: SectorId) -> Self::Scalar {
        if Self::is_tau(sector) {
            Complex64::from_polar(1.0, std::f64::consts::PI * -4.0 / 5.0)
        } else {
            Complex64::new(1.0, 0.0)
        }
    }

    // TensorKitSectors has no override either, so this is the generic
    // `frobenius_schur_phase_from_Fsymbol` (sectors.jl:461-469):
    // `sign(Fsymbol(a, dual(a), a, a, leftunit(a), rightunit(a)))`, with
    // `leftunit`/`rightunit` defaulting to `unit(a)` = vacuum (sectors.jl:
    // 139-154). For `a = τ` (self-dual) that is `Fsymbol(τ,τ,τ,τ,𝟙,𝟙) =
    // +1/φ`, whose sign is `+1`; for `a = 𝟙` it is trivially `+1`.
    fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
        Complex64::new(1.0, 0.0)
    }
}
