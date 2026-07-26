#![forbid(unsafe_code)]

//! Fusion-category *data* for TeNeT: pinned, auditable tables that become
//! `tenet-sectors` providers.
//!
//! `tenet-sectors` ships the closed-form categories whose symbols are a few
//! lines of arithmetic. That does not scale: the classification of small
//! fusion categories is a body of computed data, not a body of formulas. This
//! crate is the other half — the Rust counterpart of what CategoryData.jl is
//! to TensorKitSectors:
//!
//! ```text
//! exact algebraic source  ->  pinned decimal export  ->  Complex64 projection
//!                         ->  tenet-sectors provider  ->  F-moves and braids
//! ```
//!
//! # What you can do with it
//!
//! - Construct [`CategoryDataFibonacci`], a full multiplicity-free braided
//!   provider (`N`, duals, `F`, `R`, quantum dimensions, Frobenius–Schur
//!   phases, twists, `A`/`B`) backed by the upstream tables.
//! - Address its objects by the same one-based [`CategoryObject`] labels the
//!   upstream files use, so any value can be checked against the source by eye.
//! - Read [`CategoryDataFibonacci::PROVENANCE`], the frozen record of exactly
//!   which upstream bytes and which numeric convention produced the
//!   coefficients you are computing with.
//! - Rely on [`tenet_sectors::RuleIdentity`] to separate this provider from
//!   any other: the identity is bound to the source bytes and the projection
//!   epoch, so a changed table, gauge, object ordering, or conversion
//!   convention cannot silently reuse cached recoupling data.
//!
//! # What it deliberately is not
//!
//! - Not an exact-algebra engine. The shipped decimals are already a numerical
//!   projection of the exact Mathematica source; this crate reproduces that
//!   projection faithfully and identifiably, it does not re-derive it.
//! - Not a solver. There is no code here that solves the pentagon or hexagon
//!   equations, and no path that regenerates `F`/`R` from `N`. Those equations
//!   appear only as *consistency* tests, which catch import, indexing and
//!   gauge mistakes without ever becoming the authority for what the category
//!   is.
//! - Not online. Nothing invokes Julia, Mathematica, or the network; the
//!   tables are checked in and the reference fixtures are generated offline.
//! - Not a catalog. One category ships because one workload asked for it.
//!
//! # Relationship to the closed-form Fibonacci
//!
//! [`tenet_sectors::FibonacciFusionRule`] evaluates the same category from
//! formulas. The two providers agree physically but not bitwise — the pinned
//! decimals and the closed-form evaluation land a few ULP apart — so they have
//! *different* [`tenet_sectors::RuleIdentity`] values by design. That is not a
//! defect to paper over: each is an independent oracle for the other, and
//! sharing an identity would let recoupling data computed under one set of
//! coefficients be reused under the other.
//!
//! # Provenance
//!
//! Every upstream pin, hash, gauge statement, and the exact
//! source-to-`Complex64` conversion epoch is tabulated in this crate's
//! `references.md`. The machine-readable form is [`CategoryProvenance`].

mod provenance;
pub use provenance::{CategoryProvenance, PROJECTION_EPOCH};

mod table;
pub use table::{CategoryDataError, CategoryObject};

mod fibonacci;
pub use fibonacci::CategoryDataFibonacci;
