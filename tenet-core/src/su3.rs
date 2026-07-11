// SU(3) table-driven fusion-symbol provider (Stage B3b, #97).
//
// `Su3FusionRule` is a cheap `Arc<Su3SymbolTable>` handle over a checked-in
// table of N-/F-/R-/dim/dual/Frobenius–Schur data for the 17 `SUNIrrep{3}`
// irreps with `dim ≤ 27`. The table is generated offline by
// `tools/su3-table-gen/gen.jl` straight from SUNRepresentations (the
// TensorKitSectors sector interface), serialised to `su3_table.bin`, and
// embedded with `include_bytes!`.
//
// Hard-error boundary (fail loudly, never truncate):
//
// SU(3) fusion does not close over any finite set: `dim ≤ 27` is a cut, and a
// pair whose product escapes it (e.g. `8⊗10 ∋ 35`) is simply not representable
// here. Because `FusionRule::fusion_channels` returns a `SectorVec`, not a
// `Result`, the only physically honest options are "answer completely" or
// "refuse loudly" — returning a *partial* channel list would silently corrupt
// the fusion category (associativity, block structure, recoupling all break).
// So an escaping pair panics with a clear message, and `Su3FusionRule::covers`
// is the cheap pre-check callers use to stay inside the table. The unbounded
// successor (build CGCs in Rust, no `dim` cut) is Stage B3c.
//
// `SectorId` is the dense index into the irrep list sorted by `(dim, p, q)`, so
// vacuum `(0,0)` is id 0.

// ---------------------------------------------------------------------------
// Embedded table + one-pass loader
// ---------------------------------------------------------------------------

/// The raw table bytes (see `tools/su3-table-gen/README.md` for provenance and
/// `gen.jl` for the little-endian format).
static SU3_TABLE_BYTES: &[u8] = include_bytes!("su3_table.bin");

/// Per-irrep scalar data, indexed by dense `SectorId`.
#[derive(Clone, Debug)]
struct Su3Irrep {
    p: u8,
    q: u8,
    dim: f64,
    dual: SectorId,
    /// Frobenius–Schur phase, `±1` (a bare scalar for every fusion style — the
    /// pivotal axioms force the relevant `F` block to a single number).
    fs: f64,
}

/// The immutable SU(3) symbol table. Shared behind an `Arc`; `Su3FusionRule`
/// is a cheap clone of the handle, never of the table.
#[derive(Debug)]
pub struct Su3SymbolTable {
    irreps: Vec<Su3Irrep>,
    /// Covered `(a,b)` → sorted channel list. A pair *absent* here (with both
    /// ids in range) escapes the `dim ≤ 27` cut: `fusion_channels` (which must
    /// ENUMERATE every channel) then panics. This is the ONLY hard error at the
    /// fusion boundary.
    fusion: FxHashMap<(u8, u8), SectorVec>,
    /// `(a,b,c)` → `N(a,b,c)` for every in-table triple (derived from the R
    /// records: `rows(R(a,b,c)) == N(a,b,c)`). This covers even *escaping* pairs'
    /// in-table channels — `nsymbol(a,b,c)` asks about ONE triple, needs no
    /// enumeration, so it stays answerable where `fusion_channels(a,b)` cannot.
    nsym: FxHashMap<(u8, u8, u8), usize>,
    fsymbols: FxHashMap<[u8; 6], GenericFArray<f64>>,
    rsymbols: FxHashMap<[u8; 3], GenericRMatrix<f64>>,
    /// FNV-1a-64 of the table payload; doubles as the cache-key identity so a
    /// swapped table can never reuse another table's compiled plans.
    provenance: u64,
}

/// Minimal little-endian cursor over the embedded blob. Panics on truncation —
/// the blob is a compile-time constant, so any failure is a build/asset bug,
/// not a runtime input error.
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn u8(&mut self) -> u8 {
        let v = self.bytes[self.pos];
        self.pos += 1;
        v
    }
    fn i8(&mut self) -> i8 {
        self.u8() as i8
    }
    fn u32(&mut self) -> u32 {
        let mut b = [0u8; 4];
        b.copy_from_slice(&self.bytes[self.pos..self.pos + 4]);
        self.pos += 4;
        u32::from_le_bytes(b)
    }
    fn u64(&mut self) -> u64 {
        let mut b = [0u8; 8];
        b.copy_from_slice(&self.bytes[self.pos..self.pos + 8]);
        self.pos += 8;
        u64::from_le_bytes(b)
    }
    fn f64(&mut self) -> f64 {
        let mut b = [0u8; 8];
        b.copy_from_slice(&self.bytes[self.pos..self.pos + 8]);
        self.pos += 8;
        f64::from_le_bytes(b)
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

impl Su3SymbolTable {
    fn load() -> Self {
        let bytes = SU3_TABLE_BYTES;
        assert_eq!(&bytes[0..4], b"SU3T", "su3_table.bin: bad magic");
        let mut c = Cursor { bytes, pos: 4 };
        let version = c.u32();
        assert_eq!(version, 1, "su3_table.bin: unsupported version {version}");
        let provenance = c.u64();
        // Self-check: recompute the payload hash. Catches truncation and — the
        // Stage B2b hazard — a row/column transpose mistake in the generator or
        // reader, which would change the byte stream and so the FNV digest.
        let payload_hash = fnv1a64(&bytes[c.pos..]);
        assert_eq!(
            payload_hash, provenance,
            "su3_table.bin: payload FNV-1a mismatch (corrupt or transposed table)"
        );

        let n_irreps = c.u32() as usize;
        let mut irreps = Vec::with_capacity(n_irreps);
        for _ in 0..n_irreps {
            let p = c.u8();
            let q = c.u8();
            let dim = c.u32() as f64;
            let dual = SectorId::new(c.u8() as usize);
            let fs = c.i8() as f64;
            irreps.push(Su3Irrep { p, q, dim, dual, fs });
        }

        let n_pairs = c.u32() as usize;
        let mut fusion = FxHashMap::default();
        for _ in 0..n_pairs {
            let a = c.u8();
            let b = c.u8();
            let n_ch = c.u8() as usize;
            let mut channels: SectorVec = SectorVec::new();
            for _ in 0..n_ch {
                let cc = c.u8();
                let _nmul = c.u8(); // multiplicity comes from the R rows below
                channels.push(SectorId::new(cc as usize));
            }
            fusion.insert((a, b), channels);
        }

        let n_r = c.u32() as usize;
        let mut rsymbols = FxHashMap::default();
        let mut nsym = FxHashMap::default();
        for _ in 0..n_r {
            let a = c.u8();
            let b = c.u8();
            let cc = c.u8();
            let rows = c.u8() as usize;
            let cols = c.u8() as usize;
            let mut data = Vec::with_capacity(rows * cols);
            for _ in 0..rows * cols {
                data.push(c.f64());
            }
            // rows(R(a,b,c)) == N(a,b,c): the multiplicity of every in-table
            // triple, escaping pairs included.
            nsym.insert((a, b, cc), rows);
            rsymbols.insert([a, b, cc], GenericRMatrix::new(data, rows, cols));
        }

        let n_f = c.u32() as usize;
        let mut fsymbols = FxHashMap::default();
        for _ in 0..n_f {
            let key = [c.u8(), c.u8(), c.u8(), c.u8(), c.u8(), c.u8()];
            let s0 = c.u8() as usize;
            let s1 = c.u8() as usize;
            let s2 = c.u8() as usize;
            let s3 = c.u8() as usize;
            let len = s0 * s1 * s2 * s3;
            let mut data = Vec::with_capacity(len);
            for _ in 0..len {
                data.push(c.f64());
            }
            fsymbols.insert(key, GenericFArray::new(data, (s0, s1, s2, s3)));
        }

        Su3SymbolTable {
            irreps,
            fusion,
            nsym,
            fsymbols,
            rsymbols,
            provenance,
        }
    }

    #[inline]
    fn irrep(&self, sector: SectorId) -> &Su3Irrep {
        self.irreps.get(sector.id()).unwrap_or_else(|| {
            panic!(
                "Su3FusionRule: sector id {} is outside the dim<=27 table \
                 (0..{}); this label escaped the SU(3) hard-error boundary",
                sector.id(),
                self.irreps.len()
            )
        })
    }

    /// FNV-1a-64 of the table payload — the identity used to key cached plans.
    #[inline]
    pub fn provenance(&self) -> u64 {
        self.provenance
    }
}

/// Process-global table, parsed once on first use.
fn table() -> &'static Arc<Su3SymbolTable> {
    static TABLE: OnceLock<Arc<Su3SymbolTable>> = OnceLock::new();
    TABLE.get_or_init(|| Arc::new(Su3SymbolTable::load()))
}

// ---------------------------------------------------------------------------
// The rule handle
// ---------------------------------------------------------------------------

/// Table-driven SU(3) (`FusionStyleKind::Generic`, `dim ≤ 27`) fusion rule.
/// A cheap `Arc` handle: `Clone` copies the pointer, never the table.
#[derive(Clone, Debug)]
pub struct Su3FusionRule {
    table: Arc<Su3SymbolTable>,
}

impl Default for Su3FusionRule {
    fn default() -> Self {
        Self::new()
    }
}

impl Su3FusionRule {
    /// A handle to the process-global SU(3) table.
    pub fn new() -> Self {
        Self {
            table: Arc::clone(table()),
        }
    }

    #[inline]
    pub fn table(&self) -> &Arc<Su3SymbolTable> {
        &self.table
    }

    /// Identity of the underlying table (its payload FNV-1a-64). Embedded in the
    /// tree-transform cache key so a re-generated / swapped table never reuses
    /// another table's compiled plans.
    #[inline]
    pub fn provenance(&self) -> u64 {
        self.table.provenance
    }

    /// The dense id of the irrep with Dynkin label `(p, q)`, if it is in the
    /// `dim ≤ 27` table. `None` for out-of-table irreps.
    pub fn sector_of(&self, p: u8, q: u8) -> Option<SectorId> {
        self.table
            .irreps
            .iter()
            .position(|ir| ir.p == p && ir.q == q)
            .map(SectorId::new)
    }

    /// Dynkin label `(p, q)` of an in-table sector (for Debug / diagnostics).
    pub fn dynkin(&self, sector: SectorId) -> (u8, u8) {
        let ir = self.table.irrep(sector);
        (ir.p, ir.q)
    }

    /// Whether `left ⊗ right` is fully inside the table (both ids in range and
    /// no escaping channel). Cheap pre-check so callers avoid the
    /// `fusion_channels` hard-error panic. `covers` is *pairwise*: a deep
    /// recoupling can still reach an out-of-table internal line even when the
    /// external pair is covered — that too panics loudly at the symbol lookup.
    pub fn covers(&self, left: SectorId, right: SectorId) -> bool {
        left.id() < self.table.irreps.len()
            && right.id() < self.table.irreps.len()
            && self
                .table
                .fusion
                .contains_key(&(left.id() as u8, right.id() as u8))
    }
}

impl FusionRule for Su3FusionRule {
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        // SUNRepresentations: `BraidingStyle(::Type{<:SUNIrrep}) = Bosonic()`.
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        self.table.irrep(sector).dual
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        let key = (left.id() as u8, right.id() as u8);
        match self.table.fusion.get(&key) {
            Some(channels) => channels.clone(),
            None => {
                let (pl, ql) = self.dynkin_or_oob(left);
                let (pr, qr) = self.dynkin_or_oob(right);
                panic!(
                    "Su3FusionRule: {pl:?}⊗{qr:?} — the fusion of ({pl},{ql}) ⊗ \
                     ({pr},{qr}) escapes the dim<=27 table (a channel exceeds \
                     dim 27). This pair is outside the SU(3) hard-error boundary; \
                     use `covers(a, b)` to pre-check. Unbounded fusion is Stage B3c."
                )
            }
        }
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        // `nsymbol` asks about ONE triple — no enumeration — so it is answerable
        // for any in-table (left, right) even when `left ⊗ right` has escaping
        // channels (the recoupling engine legitimately queries in-table channels
        // of an otherwise-escaping pair, e.g. N(10̄,8,8) mid-braid). An
        // out-of-table LABEL is still a hard error (bogus sector id).
        let n = self.table.irreps.len();
        assert!(
            left.id() < n && right.id() < n,
            "Su3FusionRule::nsymbol: sector id {} or {} is outside the dim<=27 table",
            left.id(),
            right.id()
        );
        if coupled.id() >= n {
            return 0; // an out-of-table sector is never an in-table channel
        }
        self.table
            .nsym
            .get(&(left.id() as u8, right.id() as u8, coupled.id() as u8))
            .copied()
            .unwrap_or(0)
    }
}

impl Su3FusionRule {
    /// Dynkin label, or a sentinel for an out-of-range id (used only inside
    /// panic messages, so it must never itself panic).
    fn dynkin_or_oob(&self, sector: SectorId) -> (i32, i32) {
        match self.table.irreps.get(sector.id()) {
            Some(ir) => (ir.p as i32, ir.q as i32),
            None => (-1, -1),
        }
    }
}

impl GenericFusionSymbols for Su3FusionRule {
    type Scalar = f64;

    fn f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> GenericFArray<Self::Scalar> {
        let key = [
            a.id() as u8,
            b.id() as u8,
            c.id() as u8,
            d.id() as u8,
            e.id() as u8,
            f.id() as u8,
        ];
        match self.table.fsymbols.get(&key) {
            Some(block) => block.clone(),
            None => {
                // Not in the table: either an N-forbidden 6-tuple (all-zero block
                // per TensorKit) or an out-of-table label. Both are represented by
                // the shape-from-`nsymbol` zero block, EXCEPT an out-of-table label
                // makes `nsymbol` itself panic — which is the intended hard error.
                let n1 = self.nsymbol(a, b, e);
                let n2 = self.nsymbol(e, c, d);
                let n3 = self.nsymbol(b, c, f);
                let n4 = self.nsymbol(a, f, d);
                GenericFArray::new(vec![0.0; n1 * n2 * n3 * n4], (n1, n2, n3, n4))
            }
        }
    }

    fn r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> GenericRMatrix<Self::Scalar> {
        let key = [a.id() as u8, b.id() as u8, c.id() as u8];
        match self.table.rsymbols.get(&key) {
            Some(block) => block.clone(),
            None => {
                let rows = self.nsymbol(a, b, c);
                let cols = self.nsymbol(b, a, c);
                GenericRMatrix::new(vec![0.0; rows * cols], rows, cols)
            }
        }
    }
}

impl GenericRigidSymbols for Su3FusionRule {
    fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        self.table.irrep(sector).dim.sqrt()
    }

    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        1.0 / self.table.irrep(sector).dim.sqrt()
    }

    fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> Self::Scalar {
        self.table.irrep(sector).fs
    }
}
