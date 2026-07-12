// Table-driven SU(N) fusion-symbol provider (Stage B3b SU(3), #97; Stage B3c-1
// group-agnostic generalisation, #113).
//
// [`TabulatedFusionRule`] is a cheap `Arc<TabulatedSymbolTable>` handle over a
// checked-in blob of N-/F-/R-/dim/dual/Frobenius–Schur data for the `SUNIrrep{N}`
// irreps under a `dim` cut. The blob is generated offline by
// `tools/sun-table-gen/gen.jl` straight from SUNRepresentations (the
// TensorKitSectors sector interface). The SU(3) `dim ≤ 27` table is embedded
// with `include_bytes!` (`Su3FusionRule` = the alias over it); any other group
// is loaded from bytes with `from_bytes` — a new group is DATA ONLY, no Rust.
//
// Bounded-table semantics (exact or Err, never silently truncated):
//
// SU(3) fusion does not close over any finite set: `dim ≤ 27` is a cut, and a
// pair whose product escapes it (e.g. `8⊗10 ∋ 35`) is not fully representable
// here. The contract is: **block dimensions are either exactly full-SU(3) or an
// `Err` — never silently truncated.** Concretely (Option A, refute/b3b-verify):
//
// * `coupled_sector_fold` classifies each coupled-sector candidate of a leg
//   list using the v2 frontier shell (in-table channels of escaping pairs +
//   one-hop return N-symbols): `clean` sectors have their COMPLETE full-SU(3)
//   tree set inside the table and enumerate exactly; `tainted` sectors have
//   full-SU(3) trees through out-of-table inner lines (no F/R data for those)
//   and are a construction `Err`; escaped coupled candidates make full-space
//   construction an `Err` that names them. One frontier hop is tracked
//   exactly (rank ≤ 4 with a single escape); anything deeper is conservatively
//   poisoned → `Err`. Stage B3c (unbounded CGC construction) removes the cut.
// * `fusion_channels` (infallible enumeration, `SectorVec` not `Result`) still
//   panics on an escaping pair — but no public construction path can reach it:
//   space/tensor construction goes through the fold and returns `Err` first,
//   and tree transforms only run on structures the fold admitted as clean,
//   whose fold pairs are all covered (a clean space has no reachable escaping
//   pair by construction).
// * `Su3FusionRule::covers` stays as the cheap public pairwise pre-check.
//
// `SectorId` is the dense index into the irrep list sorted by `(dim, p, q)`, so
// vacuum `(0,0)` is id 0.

// ---------------------------------------------------------------------------
// Embedded table + one-pass loader
// ---------------------------------------------------------------------------

/// The raw table bytes (see `tools/su3-table-gen/README.md` for provenance and
/// `gen.jl` for the little-endian format).
static SU3_TABLE_BYTES: &[u8] = include_bytes!("su3_table.bin");
const MAX_TABLE_BYTES: usize = 64 * 1024 * 1024;
const MAX_SYMBOL_VALUES: usize = MAX_TABLE_BYTES / size_of::<f64>();
const MAX_METADATA_ENTRIES: usize = 1_000_000;
const MAX_COMPLETENESS_WORK: usize = 1_000_000;
const MAX_ASSOCIATOR_DIM: usize = 1_024;
const MAX_GRAM_WORK: usize = 100_000_000;

/// Per-irrep scalar data, indexed by dense `SectorId`.
#[derive(Clone, Debug)]
struct TabulatedIrrep {
    /// Group-agnostic Dynkin coordinates (`rank = N-1` components for `SU(N)`).
    /// SU(3) is `[p, q]`; the reader stays generic so a new group is data only.
    label: Vec<u8>,
    dim: f64,
    dual: SectorId,
    /// Frobenius–Schur phase, `±1` (a bare scalar for every fusion style — the
    /// pivotal axioms force the relevant `F` block to a single number).
    fs: f64,
}

/// The immutable tabulated symbol table for one `SU(N)` group (Stage B3c-1:
/// group-agnostic generalisation of the Stage B3b SU(3) table). Shared behind
/// an `Arc`; [`TabulatedFusionRule`] is a cheap clone of the handle, never of
/// the table. A new group is a new `su{N}_table.bin` blob — zero Rust changes.
#[derive(Debug)]
pub struct TabulatedSymbolTable {
    /// `N` of `SU(N)` (the group tag from the blob header). `rank = N - 1` is
    /// the number of Dynkin components per irrep label, recomputed where needed.
    group_n: u32,
    irreps: Vec<TabulatedIrrep>,
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
    /// v2 frontier shell: Dynkin label + dim of every out-of-table channel of
    /// an in-table pair (indexed by frontier id). Labels only — no F/R symbols
    /// exist for these, which is exactly why sectors reached through them are
    /// an `Err`, not enumerable.
    frontier: Vec<(Vec<u8>, u32)>,
    /// v2: escaping in-table pairs → (in-table channels, frontier channel ids).
    escaping: FxHashMap<(u8, u8), TabulatedEscapingPair>,
    /// v2: one-hop returns `frontier f ⊗ in-table x` → in-table channels + how
    /// far the rest of the product strays.
    one_hop: FxHashMap<(u16, u8), TabulatedOneHop>,
    /// FNV-1a-64 of the table payload; doubles as the cache-key identity so a
    /// swapped table can never reuse another table's compiled plans.
    provenance: u64,
}

#[derive(Debug)]
struct TabulatedEscapingPair {
    in_channels: SectorVec,
    frontier: Vec<u16>,
}

#[derive(Debug)]
struct TabulatedOneHop {
    /// In-table channels `(c, N(f, x, c))`.
    returns: Vec<(u8, u8)>,
    /// `f⊗x` has channels beyond table ∪ first shell.
    beyond_shell: bool,
    /// `f⊗x` has first-shell (frontier) channels.
    has_frontier: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TableError {
    Truncated { offset: usize, needed: usize, remaining: usize },
    BadMagic,
    UnsupportedVersion(u32),
    HashMismatch { expected: u64, actual: u64 },
    Overflow(&'static str),
    Invalid { section: &'static str, message: String },
    Duplicate { section: &'static str, key: String },
    MissingR([u8; 3]),
    MissingF([u8; 6]),
    TrailingBytes(usize),
}

impl fmt::Display for TableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated { offset, needed, remaining } => write!(
                f,
                "table is truncated at byte {offset}: need {needed} bytes, have {remaining}"
            ),
            Self::BadMagic => write!(f, "bad tabulated-fusion magic"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported table version {v}"),
            Self::HashMismatch { expected, actual } => write!(
                f,
                "payload FNV-1a mismatch: header {expected:#x}, computed {actual:#x}"
            ),
            Self::Overflow(context) => write!(f, "integer overflow while reading {context}"),
            Self::Invalid { section, message } => write!(f, "invalid {section}: {message}"),
            Self::Duplicate { section, key } => write!(f, "duplicate {section} record {key}"),
            Self::MissingR(key) => write!(f, "missing admissible R record {key:?}"),
            Self::MissingF(key) => write!(f, "missing admissible F record {key:?}"),
            Self::TrailingBytes(n) => write!(f, "table has {n} trailing bytes"),
        }
    }
}

impl std::error::Error for TableError {}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8], TableError> {
        let end = self.pos.checked_add(len).ok_or(TableError::Overflow("cursor offset"))?;
        let slice = self.bytes.get(self.pos..end).ok_or(TableError::Truncated {
            offset: self.pos,
            needed: len,
            remaining: self.bytes.len().saturating_sub(self.pos),
        })?;
        self.pos = end;
        Ok(slice)
    }
    fn u8(&mut self) -> Result<u8, TableError> {
        let v = self.take(1)?[0];
        Ok(v)
    }
    fn i8(&mut self) -> Result<i8, TableError> {
        Ok(self.u8()? as i8)
    }
    fn u16(&mut self) -> Result<u16, TableError> {
        let mut b = [0u8; 2];
        b.copy_from_slice(self.take(2)?);
        Ok(u16::from_le_bytes(b))
    }
    fn u32(&mut self) -> Result<u32, TableError> {
        let mut b = [0u8; 4];
        b.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(b))
    }
    fn u64(&mut self) -> Result<u64, TableError> {
        let mut b = [0u8; 8];
        b.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(b))
    }
}

/// Group-agnostic Dynkin label as `(p,q,…)` for diagnostics — the SU(3)
/// `(p,q)` shape generalises to any `rank = N-1` without a special case.
fn fmt_label(label: &[u8]) -> String {
    let inner = label
        .iter()
        .map(|c| c.to_string())
        .collect::<Vec<_>>()
        .join(",");
    format!("({inner})")
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn validate_sector_ids(
    section: &'static str,
    ids: &[u8],
    n_irreps: usize,
) -> Result<(), TableError> {
    if let Some(id) = ids.iter().copied().find(|&id| id as usize >= n_irreps) {
        return Err(TableError::Invalid {
            section,
            message: format!("sector id {id} is outside 0..{n_irreps}"),
        });
    }
    Ok(())
}

fn validate_record_count(
    section: &'static str,
    count: usize,
    minimum_record_bytes: usize,
    remaining_bytes: usize,
) -> Result<(), TableError> {
    if count > remaining_bytes / minimum_record_bytes {
        return Err(TableError::Invalid {
            section,
            message: format!("record count {count} exceeds the remaining payload"),
        });
    }
    Ok(())
}

fn consume_metadata_budget(
    section: &'static str,
    used: &mut usize,
    additional: usize,
) -> Result<(), TableError> {
    *used = used.checked_add(additional).ok_or(TableError::Overflow("metadata entry count"))?;
    if *used > MAX_METADATA_ENTRIES {
        return Err(TableError::Invalid {
            section,
            message: "metadata entry budget exceeded".into(),
        });
    }
    Ok(())
}

fn validate_symbol_completeness(
    nsym: &FxHashMap<(u8, u8, u8), usize>,
    rsymbols: &FxHashMap<[u8; 3], GenericRMatrix<f64>>,
    fsymbols: &FxHashMap<[u8; 6], GenericFArray<f64>>,
    covered_pairs: &FxHashMap<(u8, u8), SectorVec>,
) -> Result<(), TableError> {
    for (&(a, b, coupled), &rows) in nsym {
        let key = [a, b, coupled];
        let block = rsymbols.get(&key).ok_or(TableError::MissingR(key))?;
        let cols = nsym.get(&(b, a, coupled)).copied().ok_or_else(|| TableError::Invalid {
            section: "N symbols",
            message: format!("missing reverse multiplicity for {key:?}"),
        })?;
        if block.shape() != (rows, cols) {
            return Err(TableError::Invalid {
                section: "R",
                message: format!("{key:?} has shape {:?}, expected ({rows}, {cols})", block.shape()),
            });
        }
        for i in 0..cols {
            for j in 0..cols {
                let dot = (0..rows).map(|k| block.get(k, i) * block.get(k, j)).sum::<f64>();
                let expected = if i == j { 1.0 } else { 0.0 };
                if (dot - expected).abs() > 1e-10 {
                    return Err(TableError::Invalid {
                        section: "R",
                        message: format!("{key:?} is not orthogonal"),
                    });
                }
            }
        }
    }
    if rsymbols.len() != nsym.len() {
        return Err(TableError::Invalid { section: "R", message: "record set contains a forbidden key".into() });
    }

    validate_f_completeness(nsym, fsymbols)?;
    validate_f_unitarity(nsym, fsymbols, covered_pairs)?;
    Ok(())
}

fn validate_f_completeness(
    nsym: &FxHashMap<(u8, u8, u8), usize>,
    fsymbols: &FxHashMap<[u8; 6], GenericFArray<f64>>,
) -> Result<(), TableError> {
    let mut channels: FxHashMap<(u8, u8), Vec<u8>> = FxHashMap::default();
    for &(a, b, coupled) in nsym.keys() {
        channels.entry((a, b)).or_default().push(coupled);
    }
    let mut channels_by_left: FxHashMap<u8, Vec<(u8, Vec<u8>)>> = FxHashMap::default();
    for (&(left, right), coupled) in &channels {
        channels_by_left.entry(left).or_default().push((right, coupled.clone()));
    }
    let triples: Vec<(u8, u8, u8)> = nsym.keys().copied().collect();
    let mut expected_f = 0usize;
    let mut completeness_work = 0usize;
    for &(a, b, e) in &triples {
        let Some(outgoing) = channels_by_left.get(&e) else { continue };
        for &(c, ref ds) in outgoing {
            let Some(fs) = channels.get(&(b, c)) else { continue };
            for &d in ds {
                for &f in fs {
                    completeness_work = completeness_work
                        .checked_add(1)
                        .ok_or(TableError::Overflow("F completeness work"))?;
                    if completeness_work > MAX_COMPLETENESS_WORK {
                        return Err(TableError::Invalid {
                            section: "F",
                            message: "admissible-key validation work budget exceeded".into(),
                        });
                    }
                    let Some(&n4) = nsym.get(&(a, f, d)) else { continue };
                    let key = [a, b, c, d, e, f];
                    let block = fsymbols.get(&key).ok_or(TableError::MissingF(key))?;
                    let shape = (nsym[&(a, b, e)], nsym[&(e, c, d)], nsym[&(b, c, f)], n4);
                    if block.shape() != shape {
                        return Err(TableError::Invalid {
                            section: "F",
                            message: format!("{key:?} has shape {:?}, expected {shape:?}", block.shape()),
                        });
                    }
                    expected_f += 1;
                }
            }
        }
    }
    if fsymbols.len() != expected_f {
        return Err(TableError::Invalid { section: "F", message: "record set contains a forbidden or duplicate key".into() });
    }
    Ok(())
}

fn validate_f_unitarity(
    nsym: &FxHashMap<(u8, u8, u8), usize>,
    fsymbols: &FxHashMap<[u8; 6], GenericFArray<f64>>,
    covered_pairs: &FxHashMap<(u8, u8), SectorVec>,
) -> Result<(), TableError> {
    let mut groups: FxHashMap<[u8; 4], Vec<[u8; 6]>> = FxHashMap::default();
    for &key in fsymbols.keys() {
        groups.entry([key[0], key[1], key[2], key[3]]).or_default().push(key);
    }
    let mut gram_work = 0usize;
    for ([a, b, c, d], keys) in groups {
        // A partial finite-cut matrix is not an associator, so testing its Gram
        // matrix would reject valid tables rather than establish coherence.
        if !covered_pairs.contains_key(&(a, b)) || !covered_pairs.contains_key(&(b, c)) {
            continue;
        }
        let mut es: Vec<u8> = keys.iter().map(|key| key[4]).collect();
        let mut fs: Vec<u8> = keys.iter().map(|key| key[5]).collect();
        es.sort_unstable();
        es.dedup();
        fs.sort_unstable();
        fs.dedup();

        let mut rows = Vec::with_capacity(es.len());
        let mut row_count = 0usize;
        for e in es {
            let size = nsym[&(a, b, e)]
                .checked_mul(nsym[&(e, c, d)])
                .ok_or(TableError::Overflow("F associator row count"))?;
            rows.push((e, row_count, size));
            row_count = row_count.checked_add(size).ok_or(TableError::Overflow("F associator row count"))?;
        }
        let mut cols = Vec::with_capacity(fs.len());
        let mut col_count = 0usize;
        for f in fs {
            let size = nsym[&(b, c, f)]
                .checked_mul(nsym[&(a, f, d)])
                .ok_or(TableError::Overflow("F associator column count"))?;
            cols.push((f, col_count, size));
            col_count = col_count.checked_add(size).ok_or(TableError::Overflow("F associator column count"))?;
        }
        if row_count != col_count {
            return Err(TableError::Invalid {
                section: "F",
                message: format!("associator ({a},{b},{c},{d}) is {row_count}x{col_count}"),
            });
        }
        if row_count > MAX_ASSOCIATOR_DIM {
            return Err(TableError::Invalid {
                section: "F",
                message: format!("associator dimension {row_count} exceeds validation budget"),
            });
        }
        let work = row_count
            .checked_mul(row_count)
            .and_then(|n| n.checked_mul(row_count))
            .and_then(|n| n.checked_mul(2))
            .ok_or(TableError::Overflow("F Gram work"))?;
        gram_work = gram_work.checked_add(work).ok_or(TableError::Overflow("F Gram work"))?;
        if gram_work > MAX_GRAM_WORK {
            return Err(TableError::Invalid {
                section: "F",
                message: "associator validation work budget exceeded".into(),
            });
        }
        let matrix_len = row_count.checked_mul(col_count).ok_or(TableError::Overflow("F associator matrix"))?;
        if matrix_len > MAX_SYMBOL_VALUES {
            return Err(TableError::Invalid { section: "F", message: "associator validation budget exceeded".into() });
        }
        let mut matrix = vec![0.0; matrix_len];
        for key in keys {
            let block = &fsymbols[&key];
            let (_, row_offset, _) = rows.iter().find(|(e, _, _)| *e == key[4]).unwrap();
            let (_, col_offset, _) = cols.iter().find(|(f, _, _)| *f == key[5]).unwrap();
            let (s0, s1, s2, s3) = block.shape();
            for mu in 0..s0 {
                for nu in 0..s1 {
                    let row = row_offset + mu * s1 + nu;
                    for kappa in 0..s2 {
                        for lambda in 0..s3 {
                            let col = col_offset + kappa * s3 + lambda;
                            matrix[row * col_count + col] = *block.get(mu, nu, kappa, lambda);
                        }
                    }
                }
            }
        }
        for i in 0..row_count {
            for j in 0..row_count {
                let column_dot = (0..row_count)
                    .map(|k| matrix[k * row_count + i] * matrix[k * row_count + j])
                    .sum::<f64>();
                let row_dot = (0..row_count)
                    .map(|k| matrix[i * row_count + k] * matrix[j * row_count + k])
                    .sum::<f64>();
                let expected = if i == j { 1.0 } else { 0.0 };
                if (column_dot - expected).abs() > 1e-10 || (row_dot - expected).abs() > 1e-10 {
                    return Err(TableError::Invalid {
                        section: "F",
                        message: format!("associator ({a},{b},{c},{d}) is not unitary"),
                    });
                }
            }
        }
    }
    Ok(())
}

impl TabulatedSymbolTable {
    fn load_from(bytes: &[u8]) -> Result<Self, TableError> {
        if bytes.len() > MAX_TABLE_BYTES {
            return Err(TableError::Invalid {
                section: "header",
                message: format!("table exceeds the {MAX_TABLE_BYTES}-byte input budget"),
            });
        }
        if bytes.get(..4) != Some(b"TFR3") {
            return Err(TableError::BadMagic);
        }
        let mut c = Cursor { bytes, pos: 4 };
        let version = c.u32()?;
        if version != 3 {
            return Err(TableError::UnsupportedVersion(version));
        }
        let group_n = c.u32()?;
        if !(2..=256).contains(&group_n) {
            return Err(TableError::Invalid {
                section: "header",
                message: format!("SU(N) requires 2 <= N <= 256, got {group_n}"),
            });
        }
        let rank = usize::try_from(group_n - 1).map_err(|_| TableError::Overflow("rank"))?;
        let provenance = c.u64()?;
        let payload_hash = fnv1a64(&bytes[c.pos..]);
        if payload_hash != provenance {
            return Err(TableError::HashMismatch { expected: provenance, actual: payload_hash });
        }

        let n_irreps = c.u32()? as usize;
        if n_irreps == 0 || n_irreps > 256 {
            return Err(TableError::Invalid {
                section: "irreps",
                message: format!("count must be in 1..=256, got {n_irreps}"),
            });
        }
        let mut irreps = Vec::with_capacity(n_irreps);
        for _ in 0..n_irreps {
            let label: Vec<u8> = (0..rank).map(|_| c.u8()).collect::<Result<_, _>>()?;
            let dim = c.u32()? as f64;
            let dual = SectorId::new(c.u8()? as usize);
            let fs = c.i8()? as f64;
            if dim == 0.0 || !matches!(fs as i8, -1 | 1) {
                return Err(TableError::Invalid {
                    section: "irreps",
                    message: format!("dimension must be positive and FS must be +/-1 for {label:?}"),
                });
            }
            irreps.push(TabulatedIrrep {
                label,
                dim,
                dual,
                fs,
            });
        }

        if irreps[0].label.iter().any(|&x| x != 0) {
            return Err(TableError::Invalid { section: "irreps", message: "sector 0 is not vacuum".into() });
        }
        for (id, irrep) in irreps.iter().enumerate() {
            let dual = irrep.dual.id();
            if dual >= n_irreps || irreps[dual].dual.id() != id || irreps[dual].dim != irrep.dim {
                return Err(TableError::Invalid { section: "irreps", message: format!("invalid dual for sector {id}") });
            }
            if irreps[dual].label.iter().rev().copied().ne(irrep.label.iter().copied()) {
                return Err(TableError::Invalid { section: "irreps", message: format!("dual label mismatch for sector {id}") });
            }
            if irreps[..id].iter().any(|other| other.label == irrep.label) {
                return Err(TableError::Duplicate {
                    section: "irrep label",
                    key: format!("{:?}", irrep.label),
                });
            }
        }

        let n_pairs = c.u32()? as usize;
        validate_record_count("fusion", n_pairs, 3, bytes.len() - c.pos)?;
        if n_pairs > n_irreps * n_irreps {
            return Err(TableError::Invalid { section: "fusion", message: format!("too many pairs: {n_pairs}") });
        }
        let mut fusion = FxHashMap::default();
        let mut nsym = FxHashMap::default();
        let mut nsym_metadata = 0usize;
        for _ in 0..n_pairs {
            let a = c.u8()?;
            let b = c.u8()?;
            let n_ch = c.u8()? as usize;
            consume_metadata_budget("fusion", &mut nsym_metadata, n_ch)?;
            let mut channels: SectorVec = SectorVec::new();
            for _ in 0..n_ch {
                let cc = c.u8()?;
                let nmul = c.u8()? as usize;
                validate_sector_ids("fusion", &[a, b, cc], n_irreps)?;
                if nmul == 0 || nsym.insert((a, b, cc), nmul).is_some() {
                    return Err(TableError::Invalid { section: "fusion", message: format!("invalid multiplicity for ({a},{b},{cc})") });
                }
                channels.push(SectorId::new(cc as usize));
            }
            if fusion.insert((a, b), channels).is_some() {
                return Err(TableError::Duplicate { section: "fusion", key: format!("({a},{b})") });
            }
        }

        let n_r = c.u32()? as usize;
        if n_r > MAX_METADATA_ENTRIES {
            return Err(TableError::Invalid { section: "R", message: "record budget exceeded".into() });
        }
        validate_record_count("R", n_r, 5, bytes.len() - c.pos)?;
        let mut rsymbols = FxHashMap::default();
        let mut symbol_values = 0usize;
        for _ in 0..n_r {
            let a = c.u8()?;
            let b = c.u8()?;
            let cc = c.u8()?;
            validate_sector_ids("R", &[a, b, cc], n_irreps)?;
            let rows = c.u8()? as usize;
            let cols = c.u8()? as usize;
            let len = rows.checked_mul(cols).ok_or(TableError::Overflow("R shape"))?;
            symbol_values = symbol_values.checked_add(len).ok_or(TableError::Overflow("symbol value count"))?;
            if symbol_values > MAX_SYMBOL_VALUES {
                return Err(TableError::Invalid { section: "R", message: "symbol value budget exceeded".into() });
            }
            let byte_len = len.checked_mul(size_of::<f64>()).ok_or(TableError::Overflow("R byte length"))?;
            let raw = c.take(byte_len)?;
            let mut data = Vec::with_capacity(len);
            for chunk in raw.chunks_exact(size_of::<f64>()) {
                let value = f64::from_le_bytes(chunk.try_into().expect("f64 chunk has fixed width"));
                if !value.is_finite() { return Err(TableError::Invalid { section: "R", message: "non-finite coefficient".into() }); }
                data.push(value);
            }
            let key = [a, b, cc];
            if rsymbols.insert(key, GenericRMatrix::try_new(data, rows, cols).map_err(|e| TableError::Invalid { section: "R", message: e.to_string() })?).is_some() {
                return Err(TableError::Duplicate { section: "R", key: format!("{key:?}") });
            }
        }

        let n_f = c.u32()? as usize;
        if n_f > MAX_METADATA_ENTRIES {
            return Err(TableError::Invalid { section: "F", message: "record budget exceeded".into() });
        }
        validate_record_count("F", n_f, 10, bytes.len() - c.pos)?;
        let mut fsymbols = FxHashMap::default();
        for _ in 0..n_f {
            let key = [c.u8()?, c.u8()?, c.u8()?, c.u8()?, c.u8()?, c.u8()?];
            validate_sector_ids("F", &key, n_irreps)?;
            let s0 = c.u8()? as usize;
            let s1 = c.u8()? as usize;
            let s2 = c.u8()? as usize;
            let s3 = c.u8()? as usize;
            let len = s0.checked_mul(s1).and_then(|n| n.checked_mul(s2)).and_then(|n| n.checked_mul(s3)).ok_or(TableError::Overflow("F shape"))?;
            symbol_values = symbol_values.checked_add(len).ok_or(TableError::Overflow("symbol value count"))?;
            if symbol_values > MAX_SYMBOL_VALUES {
                return Err(TableError::Invalid { section: "F", message: "symbol value budget exceeded".into() });
            }
            let byte_len = len.checked_mul(size_of::<f64>()).ok_or(TableError::Overflow("F byte length"))?;
            let raw = c.take(byte_len)?;
            let mut data = Vec::with_capacity(len);
            for chunk in raw.chunks_exact(size_of::<f64>()) {
                let value = f64::from_le_bytes(chunk.try_into().expect("f64 chunk has fixed width"));
                if !value.is_finite() { return Err(TableError::Invalid { section: "F", message: "non-finite coefficient".into() }); }
                data.push(value);
            }
            if fsymbols.insert(key, GenericFArray::try_new(data, (s0, s1, s2, s3)).map_err(|e| TableError::Invalid { section: "F", message: e.to_string() })?).is_some() {
                return Err(TableError::Duplicate { section: "F", key: format!("{key:?}") });
            }
        }

        // ---- v2 frontier shell ----
        let n_frontier = c.u32()? as usize;
        if n_frontier > MAX_METADATA_ENTRIES {
            return Err(TableError::Invalid { section: "frontier", message: "record budget exceeded".into() });
        }
        validate_record_count("frontier", n_frontier, rank + 6, bytes.len() - c.pos)?;
        let mut frontier = Vec::with_capacity(n_frontier);
        let mut frontier_duals = Vec::with_capacity(n_frontier);
        for _ in 0..n_frontier {
            let label: Vec<u8> = (0..rank).map(|_| c.u8()).collect::<Result<_, _>>()?;
            let dim = c.u32()?;
            let dual_fid = c.u16()?;
            if dim == 0 || dual_fid as usize >= n_frontier {
                return Err(TableError::Invalid { section: "frontier", message: format!("invalid dimension or dual id {dual_fid}") });
            }
            frontier.push((label, dim));
            frontier_duals.push(dual_fid);
        }
        for (fid, (&dual, (label, dim))) in frontier_duals.iter().zip(&frontier).enumerate() {
            let dual = dual as usize;
            if frontier_duals[dual] as usize != fid || frontier[dual].1 != *dim || frontier[dual].0.iter().rev().copied().ne(label.iter().copied()) {
                return Err(TableError::Invalid { section: "frontier", message: format!("invalid dual for frontier {fid}") });
            }
        }

        let n_escaping = c.u32()? as usize;
        validate_record_count("escaping", n_escaping, 4, bytes.len() - c.pos)?;
        if n_escaping > n_irreps * n_irreps {
            return Err(TableError::Invalid { section: "escaping", message: format!("too many pairs: {n_escaping}") });
        }
        let mut escaping = FxHashMap::default();
        let mut escaping_metadata = 0usize;
        for _ in 0..n_escaping {
            let a = c.u8()?;
            let b = c.u8()?;
            validate_sector_ids("escaping", &[a, b], n_irreps)?;
            let n_in = c.u8()? as usize;
            consume_metadata_budget("escaping", &mut nsym_metadata, n_in)?;
            let mut in_channels: SectorVec = SectorVec::new();
            for _ in 0..n_in {
                let cc = c.u8()?;
                let nmul = c.u8()? as usize;
                validate_sector_ids("escaping", &[cc], n_irreps)?;
                if nmul == 0 || nsym.insert((a, b, cc), nmul).is_some() {
                    return Err(TableError::Invalid { section: "escaping", message: format!("invalid multiplicity for ({a},{b},{cc})") });
                }
                in_channels.push(SectorId::new(cc as usize));
            }
            let n_fr = c.u8()? as usize;
            consume_metadata_budget("escaping", &mut escaping_metadata, n_fr)?;
            if n_fr == 0 {
                return Err(TableError::Invalid { section: "escaping", message: format!("({a},{b}) has no frontier channel") });
            }
            let mut fr = Vec::with_capacity(n_fr);
            for _ in 0..n_fr {
                let fid = c.u16()?;
                if fid as usize >= n_frontier { return Err(TableError::Invalid { section: "escaping", message: format!("frontier id {fid} out of bounds") }); }
                fr.push(fid);
            }
            if escaping.insert(
                (a, b),
                TabulatedEscapingPair {
                    in_channels,
                    frontier: fr,
                },
            ).is_some() {
                return Err(TableError::Duplicate { section: "escaping", key: format!("({a},{b})") });
            }
        }
        for a in (0..n_irreps).map(|id| id as u8) {
            for b in (0..n_irreps).map(|id| id as u8) {
                if fusion.contains_key(&(a, b)) == escaping.contains_key(&(a, b)) {
                    return Err(TableError::Invalid { section: "pair partition", message: format!("({a},{b}) must occur in exactly one pair set") });
                }
            }
        }

        let n_hops = c.u32()? as usize;
        if n_hops > MAX_METADATA_ENTRIES {
            return Err(TableError::Invalid { section: "one-hop", message: "record budget exceeded".into() });
        }
        validate_record_count("one-hop", n_hops, 5, bytes.len() - c.pos)?;
        let mut one_hop = FxHashMap::default();
        let mut one_hop_metadata = 0usize;
        for _ in 0..n_hops {
            let fid = c.u16()?;
            let x = c.u8()?;
            if fid as usize >= n_frontier { return Err(TableError::Invalid { section: "one-hop", message: format!("frontier id {fid} out of bounds") }); }
            validate_sector_ids("one-hop", &[x], n_irreps)?;
            let flags = c.u8()?;
            let n_ret = c.u8()? as usize;
            consume_metadata_budget("one-hop", &mut one_hop_metadata, n_ret)?;
            let mut returns = Vec::with_capacity(n_ret);
            for _ in 0..n_ret {
                let sector = c.u8()?;
                let multiplicity = c.u8()?;
                validate_sector_ids("one-hop", &[sector], n_irreps)?;
                if multiplicity == 0 { return Err(TableError::Invalid { section: "one-hop", message: "zero return multiplicity".into() }); }
                returns.push((sector, multiplicity));
            }
            if one_hop.insert(
                (fid, x),
                TabulatedOneHop {
                    returns,
                    beyond_shell: flags & 1 != 0,
                    has_frontier: flags & 2 != 0,
                },
            ).is_some() {
                return Err(TableError::Duplicate { section: "one-hop", key: format!("({fid},{x})") });
            }
        }
        let expected_hops = n_frontier.checked_mul(n_irreps).ok_or(TableError::Overflow("one-hop count"))?;
        if one_hop.len() != expected_hops {
            return Err(TableError::Invalid { section: "one-hop", message: format!("have {}, expected {expected_hops}", one_hop.len()) });
        }
        if c.pos != bytes.len() { return Err(TableError::TrailingBytes(bytes.len() - c.pos)); }

        validate_symbol_completeness(&nsym, &rsymbols, &fsymbols, &fusion)?;

        Ok(TabulatedSymbolTable {
            group_n,
            irreps,
            fusion,
            nsym,
            fsymbols,
            rsymbols,
            frontier,
            escaping,
            one_hop,
            provenance,
        })
    }

    #[inline]
    fn irrep(&self, sector: SectorId) -> &TabulatedIrrep {
        self.irreps.get(sector.id()).unwrap_or_else(|| {
            panic!(
                "TabulatedFusionRule(SU({})): sector id {} is outside the table \
                 (0..{}); this label escaped the hard-error boundary",
                self.group_n,
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

/// Process-global SU(3) table, parsed once on first use.
fn table() -> &'static Arc<TabulatedSymbolTable> {
    static TABLE: OnceLock<Arc<TabulatedSymbolTable>> = OnceLock::new();
    TABLE.get_or_init(|| {
        Arc::new(
            TabulatedSymbolTable::load_from(SU3_TABLE_BYTES)
                .unwrap_or_else(|error| panic!("su3_table.bin: {error}")),
        )
    })
}

// ---------------------------------------------------------------------------
// The rule handle
// ---------------------------------------------------------------------------

/// Table-driven `SU(N)` (`FusionStyleKind::Generic`) fusion rule over a
/// group-agnostic tabulated symbol blob (Stage B3c-1). A cheap `Arc` handle:
/// `Clone` copies the pointer, never the table. The checked-in SU(3) table is
/// the process-global default ([`Self::new`]); any other group loads from a
/// blob with [`Self::from_bytes`] — a new group is DATA ONLY, no Rust changes.
#[derive(Clone, Debug)]
pub struct TabulatedFusionRule {
    table: Arc<TabulatedSymbolTable>,
    identity: RuleIdentity,
}

/// Thin public alias kept for the SU(3) call sites (Stage B3b): the SU(3)
/// provider is just a [`TabulatedFusionRule`] over the checked-in SU(3) blob.
pub type Su3FusionRule = TabulatedFusionRule;

impl Default for TabulatedFusionRule {
    fn default() -> Self {
        Self::new()
    }
}

impl TabulatedFusionRule {
    /// A handle to the process-global SU(3) table (the checked-in default).
    pub fn new() -> Self {
        static IDENTITY: OnceLock<RuleIdentity> = OnceLock::new();
        Self {
            table: Arc::clone(table()),
            identity: IDENTITY
                .get_or_init(RuleIdentity::new_unique::<Self>)
                .clone(),
        }
    }

    /// Parses a bounded tabulated-fusion blob and validates its structure,
    /// admissible symbol completeness, shapes, and closed-witness unitarity.
    /// Category provenance and gauge identity still require comparison with the
    /// generator's independent SUNRepresentations oracle.
    pub fn try_from_bytes(bytes: &[u8], _name: &'static str) -> Result<Self, TableError> {
        Ok(Self {
            table: Arc::new(TabulatedSymbolTable::load_from(bytes)?),
            identity: RuleIdentity::new_unique::<Self>(),
        })
    }

    /// `N` of the `SU(N)` group this rule tabulates.
    #[inline]
    pub fn group_n(&self) -> u32 {
        self.table.group_n
    }

    #[inline]
    pub fn table(&self) -> &Arc<TabulatedSymbolTable> {
        &self.table
    }

    /// Identity of the underlying table (its payload FNV-1a-64). Embedded in the
    /// tree-transform cache key so a re-generated / swapped table never reuses
    /// another table's compiled plans.
    #[inline]
    pub fn provenance(&self) -> u64 {
        self.table.provenance
    }

    /// The dense id of the irrep with the given Dynkin label, if it is in the
    /// table. `None` for out-of-table irreps. Group-agnostic: `label` is the
    /// full `rank = N-1` coordinate list.
    pub fn sector_of_label(&self, label: &[u8]) -> Option<SectorId> {
        self.table
            .irreps
            .iter()
            .position(|ir| ir.label == label)
            .map(SectorId::new)
    }

    /// SU(3) convenience: the dense id of the irrep with Dynkin label `(p, q)`.
    /// A thin wrapper over [`Self::sector_of_label`] for the rank-2 SU(3) table.
    pub fn sector_of(&self, p: u8, q: u8) -> Option<SectorId> {
        self.sector_of_label(&[p, q])
    }

    /// Dynkin label of an in-table sector (for Debug / diagnostics), the full
    /// `rank = N-1` coordinate list.
    pub fn label(&self, sector: SectorId) -> &[u8] {
        &self.table.irrep(sector).label
    }

    /// SU(3) convenience: Dynkin label `(p, q)` of an in-table sector.
    pub fn dynkin(&self, sector: SectorId) -> (u8, u8) {
        let label = &self.table.irrep(sector).label;
        (label[0], label[1])
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

impl FusionRule for TabulatedFusionRule {
    fn rule_identity(&self) -> RuleIdentity {
        self.identity.clone()
    }

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
                let (l, r) = (self.label_or_oob(left), self.label_or_oob(right));
                let n = self.table.group_n;
                panic!(
                    "TabulatedFusionRule(SU({n})): the fusion of {l} ⊗ {r} escapes \
                     the table (a channel exceeds the dim cut). This pair is outside \
                     the hard-error boundary; use `covers(a, b)` to pre-check. \
                     Unbounded fusion is Stage B3c."
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
            "TabulatedFusionRule::nsymbol: sector id {} or {} is outside the table",
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

    fn fusion_channels_in_table(&self, left: SectorId, right: SectorId) -> SectorVec {
        // In-table channels of ANY in-table pair, covered or escaping. Safe to
        // use only where out-of-table branches provably vanish — i.e. on trees
        // of `clean` sectors (see the trait doc): a clean sector has no
        // full-SU(3) tree through an out-of-table line, so the skipped frontier
        // channels are dead branches, not truncation.
        let key = (left.id() as u8, right.id() as u8);
        if let Some(channels) = self.table.fusion.get(&key) {
            return channels.clone();
        }
        if let Some(esc) = self.table.escaping.get(&key) {
            return esc.in_channels.clone();
        }
        let (l, r) = (self.label_or_oob(left), self.label_or_oob(right));
        panic!(
            "TabulatedFusionRule(SU({})): sector pair {l} ⊗ {r} is not in the \
             table (invalid sector id)",
            self.table.group_n
        )
    }

    fn coupled_sector_fold(&self, effective: &[SectorId]) -> CoupledSectorFold {
        // Escape-tracking forward fold (Option A, refute/b3b-verify). States:
        //   clean    — reached only through in-table lines; their trees are
        //              exactly the full-SU(3) set (enumerable);
        //   tainted  — in-table, but some full-SU(3) tree reaches them through
        //              an out-of-table line (one-hop return) → must be Err;
        //   frontier — out-of-table intermediate states, folded via the v2
        //              one-hop table for exactly one more step.
        // A frontier surviving PAST one hop (or leaving the first shell) sets
        // `poisoned`: the clean/tainted split is unknown → everything Err.
        // ponytail: exact for one frontier hop (rank<=4 single escape, all the
        // physics cases B3b targets); deeper folds are conservative Err until
        // B3c lifts the dim cut.
        use std::collections::BTreeSet;
        let t = &self.table;
        let mut clean: BTreeSet<u8> = BTreeSet::new();
        let mut tainted: BTreeSet<u8> = BTreeSet::new();
        let mut frontier: BTreeSet<u16> = BTreeSet::new();
        let mut escaped: BTreeSet<u16> = BTreeSet::new();
        let mut unknown_escape = false;
        let mut poisoned = false;

        match effective.first() {
            None => {
                clean.insert(self.vacuum().id() as u8);
            }
            Some(&first) => {
                let _ = t.irrep(first); // range check, panics on bogus id
                clean.insert(first.id() as u8);
            }
        }

        for (step, &x) in effective.iter().enumerate().skip(1) {
            let _ = t.irrep(x); // range check
            let xg = x.id() as u8;
            let is_last = step == effective.len() - 1;
            let mut new_clean: BTreeSet<u8> = BTreeSet::new();
            let mut new_tainted: BTreeSet<u8> = BTreeSet::new();
            let mut new_frontier: BTreeSet<u16> = BTreeSet::new();

            let fold_in_table = |s: u8, out: &mut BTreeSet<u8>,
                                     new_frontier: &mut BTreeSet<u16>,
                                     escaped: &mut BTreeSet<u16>| {
                if let Some(channels) = t.fusion.get(&(s, xg)) {
                    out.extend(channels.iter().map(|c| c.id() as u8));
                } else if let Some(esc) = t.escaping.get(&(s, xg)) {
                    out.extend(esc.in_channels.iter().map(|c| c.id() as u8));
                    for &fid in &esc.frontier {
                        if is_last {
                            escaped.insert(fid); // out-of-table coupled candidate
                        } else {
                            new_frontier.insert(fid);
                        }
                    }
                } else {
                    unreachable!("every in-table pair is covered or escaping");
                }
            };
            for &s in &clean {
                fold_in_table(s, &mut new_clean, &mut new_frontier, &mut escaped);
            }
            for &s in &tainted {
                // taint propagates: trees through a tainted state stay incomplete.
                fold_in_table(s, &mut new_tainted, &mut new_frontier, &mut escaped);
            }
            for &f in &frontier {
                let hop = t.one_hop.get(&(f, xg)).unwrap_or_else(|| {
                    unreachable!("one-hop table covers every (frontier, in-table) pair")
                });
                // In-table returns: candidates whose enumeration would need the
                // out-of-table intermediate `f` — incomplete with our F/R data.
                new_tainted.extend(hop.returns.iter().map(|&(c, _)| c));
                if is_last {
                    if hop.beyond_shell || hop.has_frontier {
                        // Out-of-table coupled candidates with unrecorded labels.
                        unknown_escape = true;
                    }
                } else if hop.beyond_shell || hop.has_frontier {
                    // A frontier product would have to fold AGAIN; the one-hop
                    // table doesn't identify those states → conservative.
                    poisoned = true;
                }
            }

            clean = new_clean;
            tainted = new_tainted;
            frontier = new_frontier;
            if poisoned {
                break;
            }
        }

        if poisoned {
            // Split unknown: report every known candidate as tainted.
            tainted.extend(clean.iter().copied());
            clean.clear();
        } else {
            // A sector reached both cleanly and through a frontier is incomplete.
            clean.retain(|s| !tainted.contains(s));
        }

        let mut out_of_table: Vec<String> = escaped
            .iter()
            .map(|&fid| {
                let (label, dim) = &t.frontier[fid as usize];
                format!("{} dim {dim}", fmt_label(label))
            })
            .collect();
        if unknown_escape {
            out_of_table.push("(beyond one-hop frontier products)".to_string());
        }

        CoupledSectorFold {
            clean: clean.into_iter().map(|s| SectorId::new(s as usize)).collect(),
            tainted: tainted
                .into_iter()
                .map(|s| SectorId::new(s as usize))
                .collect(),
            out_of_table,
            poisoned,
        }
    }
}

impl TabulatedFusionRule {
    /// Dynkin label as a debug string, or a sentinel for an out-of-range id
    /// (used only inside panic messages, so it must never itself panic).
    /// Group-agnostic: prints the full `rank = N-1` coordinate list.
    fn label_or_oob(&self, sector: SectorId) -> String {
        match self.table.irreps.get(sector.id()) {
            Some(ir) => fmt_label(&ir.label),
            None => "<out-of-table>".to_string(),
        }
    }

    fn table_key(&self, sector: SectorId) -> u8 {
        self.table.irrep(sector);
        u8::try_from(sector.id()).expect("validated table sector must fit in u8")
    }
}

impl GenericFusionSymbols for TabulatedFusionRule {
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
            self.table_key(a),
            self.table_key(b),
            self.table_key(c),
            self.table_key(d),
            self.table_key(e),
            self.table_key(f),
        ];
        match self.table.fsymbols.get(&key) {
            Some(block) => block.clone(),
            None => {
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
        let key = [self.table_key(a), self.table_key(b), self.table_key(c)];
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

impl GenericRigidSymbols for TabulatedFusionRule {
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
