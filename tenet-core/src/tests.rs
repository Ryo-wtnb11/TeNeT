/// Test-only synchronization for tenet-core's process-global intern tables
/// (block-structure content/arc tables, the hom-space intern table).
///
/// Why-not (the alternatives this replaces): a per-test-file `#[serial]`
/// dependency would serialize this whole (large) test module for the sake of
/// a handful of tests; making the tables test-scoped (thread-local, or wiped
/// between tests) would stop exercising the process-global design these
/// tables actually ship with, and would hide the exact "concurrent
/// reset/flood lands between two reads" bugs this suite exists to catch
/// (see tenet-tensors #169, #172 for the shape of the bug).
///
/// So: one process-wide `Mutex`, taken by every test that either mutates
/// shared intern-table state (`reset_core_intern_tables`, LRU-cap floods) or
/// asserts on it (`Arc::ptr_eq` of interned values, table lengths, content
/// ids). Poison-tolerant: a panicking test must not cascade spurious
/// failures onto every other test sharing the lock.
pub(crate) mod test_support {
    pub(crate) static CACHE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
}

mod tests {
    use super::*;

    // Shared fixtures and helper fusion rules used across more than one of
    // the per-responsibility test files below (#1590). Kept flat here
    // (rather than in a `mod support`) so every child module can see them
    // through the ordinary private-item-visible-to-descendants rule, the
    // same way they saw each other before this split.
    include!("tests/support.rs");

    include!("tests/storage.rs");
    include!("tests/sector.rs");
    include!("tests/fusion_space.rs");
    include!("tests/fusion_tree.rs");
    include!("tests/block_structure.rs");
    include!("tests/tensor_map.rs");
    include!("tests/error.rs");
}
