//! Parsing and structural validation of the pinned upstream text tables.
//!
//! The record layouts are upstream's (`references.md`, "Source record
//! layouts"). Everything in this module is the *import* boundary: it turns
//! untrusted-shaped text into an indexable table or an error, and it validates
//! exactly the properties that are exact integer facts about `N`. Coherence
//! equations (unitarity, pentagon, hexagon) deliberately live in the crate's
//! tests instead — see the module note in `lib.rs`.

use num_complex::Complex64;

/// A category object, in the one-based labelling the upstream files use.
///
/// This is the crate's public sector label: `CategoryObject(1)` is the unit,
/// and the numbering is exactly the numbering in the pinned text files, so a
/// label can be checked against upstream by eye. The zero-based
/// [`tenet_sectors::SectorId`] used inside the fusion engine is an internal
/// representation; [`tenet_sectors::SectorCodec`] is the only bridge.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct CategoryObject(pub u16);

impl CategoryObject {
    /// The one-based label as written in the source files.
    #[inline]
    pub const fn label(self) -> u16 {
        self.0
    }
}

impl core::fmt::Display for CategoryObject {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "object {}", self.0)
    }
}

/// Why a pinned table could not be turned into a provider.
///
/// Every variant names a defect in the embedded data or in this crate's
/// reading of it, never a caller mistake: the tables ship with the crate, so
/// in a released build these are unreachable and the tests are what keep them
/// so.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CategoryDataError {
    /// A record did not have the field count its layout requires.
    FieldCount {
        /// Which table the record came from.
        table: &'static str,
        /// One-based line number within that table.
        line: usize,
        /// Field count the layout requires.
        expected: usize,
        /// Field count actually found.
        found: usize,
    },
    /// A field could not be read as the number its layout requires.
    Field {
        /// Which table the record came from.
        table: &'static str,
        /// One-based line number within that table.
        line: usize,
        /// The offending field, verbatim.
        field: String,
    },
    /// An object label fell outside `1..=rank`.
    ObjectRange {
        /// Which table the record came from.
        table: &'static str,
        /// One-based line number within that table.
        line: usize,
        /// The offending label.
        label: i64,
    },
    /// A multiplicity index was not `1`, so the table is not multiplicity-free.
    NotMultiplicityFree {
        /// Which table the record came from.
        table: &'static str,
        /// One-based line number within that table.
        line: usize,
    },
    /// Two records addressed the same entry.
    DuplicateEntry {
        /// Which table the record came from.
        table: &'static str,
        /// One-based line number of the second record.
        line: usize,
    },
    /// A coefficient was infinite or NaN.
    NonFinite {
        /// Which table the record came from.
        table: &'static str,
        /// One-based line number within that table.
        line: usize,
    },
    /// `N` violates an exact axiom: the unit laws, or unique involutive duals.
    FusionAxiom {
        /// Which axiom failed, and for which object.
        detail: String,
    },
}

impl core::fmt::Display for CategoryDataError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::FieldCount {
                table,
                line,
                expected,
                found,
            } => write!(
                formatter,
                "{table} line {line}: expected {expected} fields, found {found}"
            ),
            Self::Field { table, line, field } => {
                write!(formatter, "{table} line {line}: unreadable field {field:?}")
            }
            Self::ObjectRange { table, line, label } => {
                write!(
                    formatter,
                    "{table} line {line}: object label {label} out of range"
                )
            }
            Self::NotMultiplicityFree { table, line } => {
                write!(
                    formatter,
                    "{table} line {line}: multiplicity index is not 1"
                )
            }
            Self::DuplicateEntry { table, line } => {
                write!(formatter, "{table} line {line}: duplicate entry")
            }
            Self::NonFinite { table, line } => {
                write!(formatter, "{table} line {line}: non-finite coefficient")
            }
            Self::FusionAxiom { detail } => write!(formatter, "fusion axiom violated: {detail}"),
        }
    }
}

impl std::error::Error for CategoryDataError {}

/// Dense `N`/`F`/`R` tables for one multiplicity-free category, zero-based.
///
/// Storage is dense over `rank^6` `F` entries. That is O(rank^6) in the number
/// of *distinct symbol arguments*, i.e. exactly the size of the object the
/// engine queries, and for the rank ≤ 6 categories this crate targets it is at
/// most 46656 complex numbers. Why not a sparse map: every lookup would pay a
/// hash for a table that fits in L2 anyway, and the engine calls
/// `f_symbol_scalar` on the recoupling path.
#[derive(Clone, Debug)]
pub(crate) struct MultiplicityFreeTable {
    rank: usize,
    nsym: Vec<u32>,
    fsym: Vec<Complex64>,
    rsym: Vec<Complex64>,
    duals: Vec<usize>,
}

/// Reads whitespace-separated fields, rejecting anything the layout forbids.
struct Record<'a> {
    table: &'static str,
    line: usize,
    fields: Vec<&'a str>,
    rank: usize,
}

impl Record<'_> {
    /// One-based source label -> zero-based index, range-checked.
    fn object(&self, position: usize) -> Result<usize, CategoryDataError> {
        let text = self.fields[position];
        let label: i64 = text.parse().map_err(|_| CategoryDataError::Field {
            table: self.table,
            line: self.line,
            field: text.to_owned(),
        })?;
        if label < 1 || label > self.rank as i64 {
            return Err(CategoryDataError::ObjectRange {
                table: self.table,
                line: self.line,
                label,
            });
        }
        Ok(label as usize - 1)
    }

    /// A multiplicity index, which must be exactly `1` in this crate's tier.
    fn unit_multiplicity(&self, position: usize) -> Result<(), CategoryDataError> {
        if self.fields[position] == "1" {
            Ok(())
        } else {
            Err(CategoryDataError::NotMultiplicityFree {
                table: self.table,
                line: self.line,
            })
        }
    }

    fn nonnegative(&self, position: usize) -> Result<u32, CategoryDataError> {
        let text = self.fields[position];
        text.parse().map_err(|_| CategoryDataError::Field {
            table: self.table,
            line: self.line,
            field: text.to_owned(),
        })
    }

    /// The projection epoch's step 2 and 3: two correctly-rounded `f64`
    /// conversions paired into a `Complex64`.
    fn coefficient(&self, position: usize) -> Result<Complex64, CategoryDataError> {
        let mut parts = [0.0f64; 2];
        for (slot, text) in parts.iter_mut().zip(&self.fields[position..position + 2]) {
            *slot = text.parse().map_err(|_| CategoryDataError::Field {
                table: self.table,
                line: self.line,
                field: (*text).to_owned(),
            })?;
        }
        if !parts.iter().all(|value| value.is_finite()) {
            return Err(CategoryDataError::NonFinite {
                table: self.table,
                line: self.line,
            });
        }
        Ok(Complex64::new(parts[0], parts[1]))
    }
}

/// Splits `text` into records, skipping blank lines, and checks field arity.
fn records<'a>(
    table: &'static str,
    text: &'a str,
    rank: usize,
    arity: usize,
) -> impl Iterator<Item = Result<Record<'a>, CategoryDataError>> {
    text.lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(move |(index, line)| {
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() != arity {
                return Err(CategoryDataError::FieldCount {
                    table,
                    line: index + 1,
                    expected: arity,
                    found: fields.len(),
                });
            }
            Ok(Record {
                table,
                line: index + 1,
                fields,
                rank,
            })
        })
}

impl MultiplicityFreeTable {
    /// Parses the three pinned tables of a rank-`rank` multiplicity-free
    /// category and validates the exact `N` axioms.
    ///
    /// `n_text`/`f_text`/`r_text` are the upstream files verbatim. Records
    /// absent from a file are exact zeros, per the projection epoch.
    pub(crate) fn parse(
        rank: usize,
        n_text: &str,
        f_text: &str,
        r_text: &str,
    ) -> Result<Self, CategoryDataError> {
        let mut nsym = vec![0u32; rank.pow(3)];
        let mut seen = vec![false; rank.pow(3)];
        for record in records("Nsymbols", n_text, rank, 4) {
            let record = record?;
            let (a, b, c) = (record.object(0)?, record.object(1)?, record.object(2)?);
            let index = (a * rank + b) * rank + c;
            if core::mem::replace(&mut seen[index], true) {
                return Err(CategoryDataError::DuplicateEntry {
                    table: "Nsymbols",
                    line: record.line,
                });
            }
            nsym[index] = record.nonnegative(3)?;
        }

        let mut fsym = vec![Complex64::new(0.0, 0.0); rank.pow(6)];
        let mut seen = vec![false; rank.pow(6)];
        for record in records("Fsymbols", f_text, rank, 12) {
            let record = record?;
            // Layout: a b c d α e β μ f ν Re Im. The four multiplicity slots
            // are 4, 6, 7, 9 and must all be 1 in the multiplicity-free tier.
            for position in [4, 6, 7, 9] {
                record.unit_multiplicity(position)?;
            }
            let labels = [0usize, 1, 2, 3, 5, 8].map(|position| record.object(position));
            let mut index = 0usize;
            for label in labels {
                index = index * rank + label?;
            }
            if core::mem::replace(&mut seen[index], true) {
                return Err(CategoryDataError::DuplicateEntry {
                    table: "Fsymbols",
                    line: record.line,
                });
            }
            fsym[index] = record.coefficient(10)?;
        }

        let mut rsym = vec![Complex64::new(0.0, 0.0); rank.pow(3)];
        let mut seen = vec![false; rank.pow(3)];
        for record in records("Rsymbols", r_text, rank, 7) {
            let record = record?;
            // Layout: a b c α μ Re Im.
            for position in [3, 4] {
                record.unit_multiplicity(position)?;
            }
            let (a, b, c) = (record.object(0)?, record.object(1)?, record.object(2)?);
            let index = (a * rank + b) * rank + c;
            if core::mem::replace(&mut seen[index], true) {
                return Err(CategoryDataError::DuplicateEntry {
                    table: "Rsymbols",
                    line: record.line,
                });
            }
            rsym[index] = record.coefficient(5)?;
        }

        let duals = derive_duals(rank, &nsym)?;
        let table = Self {
            rank,
            nsym,
            fsym,
            rsym,
            duals,
        };
        table.check_unit_laws()?;
        Ok(table)
    }

    #[inline]
    pub(crate) fn rank(&self) -> usize {
        self.rank
    }

    /// `N^{ab}_c`, or `0` outside the table.
    #[inline]
    pub(crate) fn nsymbol(&self, a: usize, b: usize, c: usize) -> u32 {
        self.nsym[(a * self.rank + b) * self.rank + c]
    }

    /// `F^{abc}_d[e, f]`, or `0` outside the table.
    #[inline]
    pub(crate) fn fsymbol(
        &self,
        a: usize,
        b: usize,
        c: usize,
        d: usize,
        e: usize,
        f: usize,
    ) -> Complex64 {
        let index = ((((a * self.rank + b) * self.rank + c) * self.rank + d) * self.rank + e)
            * self.rank
            + f;
        self.fsym[index]
    }

    /// `R^{ab}_c`, or `0` outside the table.
    #[inline]
    pub(crate) fn rsymbol(&self, a: usize, b: usize, c: usize) -> Complex64 {
        self.rsym[(a * self.rank + b) * self.rank + c]
    }

    #[inline]
    pub(crate) fn dual(&self, a: usize) -> usize {
        self.duals[a]
    }

    /// Ascending channels of `a ⊗ b`. Ascending because the derived twist sums
    /// over this order and must reproduce TensorKitSectors' `a ⊗ b` iteration,
    /// which filters `values(I)` in label order.
    pub(crate) fn channels(&self, a: usize, b: usize) -> impl Iterator<Item = usize> + '_ {
        (0..self.rank).filter(move |&c| self.nsymbol(a, b, c) > 0)
    }
}

/// `dual(a)` is the unique `b` with `N^{ab}_1 = 1`, the derivation
/// CategoryData.jl uses (`src/objects.jl`, `TensorKitSectors.dual`).
///
/// Why derive rather than read the `selfduality` type parameter: the parameter
/// encodes how many objects are *non*-self-dual, so trusting it would make the
/// dual depend on a file name rather than on the shipped fusion data.
fn derive_duals(rank: usize, nsym: &[u32]) -> Result<Vec<usize>, CategoryDataError> {
    let nsymbol = |a: usize, b: usize, c: usize| nsym[(a * rank + b) * rank + c];
    let mut duals = Vec::with_capacity(rank);
    for a in 0..rank {
        let mut found = (0..rank).filter(|&b| nsymbol(a, b, 0) == 1);
        let dual = found.next().ok_or_else(|| CategoryDataError::FusionAxiom {
            detail: format!("object {} has no dual", a + 1),
        })?;
        if found.next().is_some() {
            return Err(CategoryDataError::FusionAxiom {
                detail: format!("object {} has more than one dual", a + 1),
            });
        }
        duals.push(dual);
    }
    for a in 0..rank {
        if duals[duals[a]] != a {
            return Err(CategoryDataError::FusionAxiom {
                detail: format!("dual is not involutive at object {}", a + 1),
            });
        }
    }
    Ok(duals)
}

impl MultiplicityFreeTable {
    /// Object 1 is the unit and acts as one on both sides.
    ///
    /// These are exact integer facts, so they are checked here rather than in
    /// tests: they are what makes the zero-based `SectorId(0)` vacuum and the
    /// `CanonicalUnitFusionRule` marker true statements about *this* table.
    fn check_unit_laws(&self) -> Result<(), CategoryDataError> {
        for a in 0..self.rank {
            for b in 0..self.rank {
                let expected = u32::from(a == b);
                for (side, actual) in [
                    ("left", self.nsymbol(0, a, b)),
                    ("right", self.nsymbol(a, 0, b)),
                ] {
                    if actual != expected {
                        return Err(CategoryDataError::FusionAxiom {
                            detail: format!(
                                "{side} unit law: N({}, {}) -> {} is {actual}, expected {expected}",
                                a + 1,
                                b + 1,
                                b + 1
                            ),
                        });
                    }
                }
            }
        }
        if self.duals[0] != 0 {
            return Err(CategoryDataError::FusionAxiom {
                detail: "unit object is not self-dual".to_owned(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A minimal well-formed rank-1 (trivial) category, used to exercise the
    // rejection paths without dragging in the shipped Fibonacci tables.
    const TRIVIAL_N: &str = "1 1 1 1\n";
    const TRIVIAL_F: &str = "1 1 1 1 1 1 1 1 1 1 1.0 0\n";
    const TRIVIAL_R: &str = "1 1 1 1 1 1.0 0\n";

    fn parse_trivial(
        n: &str,
        f: &str,
        r: &str,
    ) -> Result<MultiplicityFreeTable, CategoryDataError> {
        MultiplicityFreeTable::parse(1, n, f, r)
    }

    #[test]
    fn parses_a_trivial_category_and_skips_blank_lines() {
        let table = parse_trivial("\n1 1 1 1\n\n", TRIVIAL_F, TRIVIAL_R).expect("valid");
        assert_eq!(table.rank(), 1);
        assert_eq!(table.nsymbol(0, 0, 0), 1);
        assert_eq!(table.fsymbol(0, 0, 0, 0, 0, 0), Complex64::new(1.0, 0.0));
        assert_eq!(table.rsymbol(0, 0, 0), Complex64::new(1.0, 0.0));
        assert_eq!(table.dual(0), 0);
        assert_eq!(table.channels(0, 0).collect::<Vec<_>>(), vec![0]);
    }

    #[test]
    fn rejects_wrong_field_counts() {
        assert!(matches!(
            parse_trivial("1 1 1\n", TRIVIAL_F, TRIVIAL_R),
            Err(CategoryDataError::FieldCount {
                table: "Nsymbols",
                line: 1,
                expected: 4,
                found: 3
            })
        ));
    }

    #[test]
    fn rejects_out_of_range_objects() {
        assert!(matches!(
            parse_trivial("1 1 2 1\n", TRIVIAL_F, TRIVIAL_R),
            Err(CategoryDataError::ObjectRange { label: 2, .. })
        ));
        assert!(matches!(
            parse_trivial("1 1 0 1\n", TRIVIAL_F, TRIVIAL_R),
            Err(CategoryDataError::ObjectRange { label: 0, .. })
        ));
    }

    #[test]
    fn rejects_duplicate_entries() {
        assert!(matches!(
            parse_trivial("1 1 1 1\n1 1 1 1\n", TRIVIAL_F, TRIVIAL_R),
            Err(CategoryDataError::DuplicateEntry {
                table: "Nsymbols",
                line: 2
            })
        ));
    }

    #[test]
    fn rejects_outer_multiplicity() {
        assert!(matches!(
            parse_trivial(TRIVIAL_N, "1 1 1 1 1 1 1 2 1 1 1.0 0\n", TRIVIAL_R),
            Err(CategoryDataError::NotMultiplicityFree {
                table: "Fsymbols",
                ..
            })
        ));
        assert!(matches!(
            parse_trivial(TRIVIAL_N, TRIVIAL_F, "1 1 1 1 2 1.0 0\n"),
            Err(CategoryDataError::NotMultiplicityFree {
                table: "Rsymbols",
                ..
            })
        ));
    }

    #[test]
    fn rejects_non_finite_and_unreadable_coefficients() {
        // `TRIVIAL_R`, not `TRIVIAL_F`, as the R text: passing a 12-field F
        // record to the 7-field R parser would make this pass on an arity
        // error the moment the parse order changed.
        assert!(matches!(
            parse_trivial(TRIVIAL_N, "1 1 1 1 1 1 1 1 1 1 inf 0\n", TRIVIAL_R),
            Err(CategoryDataError::NonFinite {
                table: "Fsymbols",
                ..
            })
        ));
        assert!(matches!(
            parse_trivial(TRIVIAL_N, "1 1 1 1 1 1 1 1 1 1 zzz 0\n", TRIVIAL_R),
            Err(CategoryDataError::Field {
                table: "Fsymbols",
                ..
            })
        ));
    }

    #[test]
    fn rejects_missing_and_ambiguous_duals() {
        // Rank 2 with no object fusing to the unit at all.
        let broken = MultiplicityFreeTable::parse(2, "1 1 1 1\n1 2 2 1\n2 1 2 1\n", "", "");
        assert!(matches!(broken, Err(CategoryDataError::FusionAxiom { .. })));
    }

    #[test]
    fn rejects_a_broken_unit_law() {
        // Rank 2 where the unit does not act as one on object 2.
        let broken = MultiplicityFreeTable::parse(2, "1 1 1 1\n2 2 1 1\n", "", "");
        assert!(matches!(broken, Err(CategoryDataError::FusionAxiom { .. })));
    }
}
