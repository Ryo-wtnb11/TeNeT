//! Test-only synchronization for tenet-tensors' process-global caches
//! (the operation-cache registry, its chained `tenet-core` intern tables,
//! and the scratch-space structure cache).
//!
//! Why-not (the alternatives this replaces):
//! - Per-test-file `#[serial]` (a crate dependency) serializes every test in
//!   a module even though only a handful touch process-global state — most
//!   of a file's tests would pay a needless throughput tax.
//! - Making the caches test-scoped (thread-local, or reset before/after each
//!   test) would stop testing what the process-global design actually does
//!   in production (one process, many callers sharing one cache) and would
//!   hide exactly the "a concurrent reset lands between two reads" bugs this
//!   suite exists to catch.
//!
//! So: one process-wide `Mutex`, taken by every test that either mutates
//! shared cache state (`reset_global_operation_caches`, LRU-cap floods) or
//! asserts on it (`Arc::ptr_eq` of cached values, intern-table lengths/ids).
//! Both species must serialize against each other, not just against their
//! own kind — a reader racing an unlocked resetter is exactly the bug class
//! this lock exists to close (see #169, #172).
//!
//! Poison-tolerant: a panicking test must not poison the mutex and cascade
//! spurious failures onto every other test sharing it, so callers use
//! `.lock().unwrap_or_else(|e| e.into_inner())` rather than `.unwrap()`.
#[cfg(test)]
pub(crate) static CACHE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
