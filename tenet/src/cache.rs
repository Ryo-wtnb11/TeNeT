//! Process-global structure caches: the one public control of TeNeT's
//! retained structural data.
//!
//! Three caches, each a pure function of its key and each byte-bounded on its
//! own budget (64 MiB by default), shared by every
//! [`Runtime`](crate::typed::Runtime) of the process:
//!
//! - fusion-tree keys and sector structure per HomSpace sector signature,
//!   [`StructureCacheKind::SectorStructure`] (TensorKit `sectorstructure`);
//! - complete block structure per HomSpace,
//!   [`StructureCacheKind::DegeneracyStructure`] (TensorKit
//!   `degeneracystructure`);
//! - completed permute/braid/transpose transformers,
//!   [`StructureCacheKind::CompletedTreeTransformer`] (TensorKit
//!   `treetransposer` / `treebraider`).
//!
//! This mirrors TensorKit's `GLOBAL_CACHES` with `empty_globalcaches!` and
//! `global_cache_info` (`caches.jl:1-11` @cfaa073). Each Runtime's
//! categorical-coefficient tiers (plans and per-group recoupling) are not
//! here yet; they stay per Runtime, at a fixed 64 MiB per tier, until
//! #2014-4.
//!
//! A cache hit is interchangeable with a rebuild, so clearing or sizing these
//! changes cost, never results.

pub use tenet_core::{StructureCacheInfo, StructureCacheKind};

/// Snapshots of every structure cache, in [`StructureCacheKind::ALL`] order.
/// Each is sampled separately, not as one atomic snapshot.
pub fn stats() -> Vec<StructureCacheInfo> {
    tenet_core::structure_cache_infos()
}

/// Clears every process-global structure cache (sector, degeneracy,
/// completed tree transformer) for all Runtimes. Semantic data only: no
/// Runtime coefficient tier, device executor, scratch or racah symbol cache
/// is touched. Builds in flight across the clear return their result
/// unpublished.
///
/// Live tensors stay valid: their structures and transformers are held by
/// value, and structure ids are never reused.
pub fn clear() {
    tenet_core::clear_structure_caches();
}

/// Sets the byte budget of each listed cache; entries beyond a lowered
/// budget are evicted. Owned by the application: library code never calls it.
///
/// A completed transformer is published only while every block structure it
/// is keyed on is resident in, or was admitted to, the
/// [`StructureCacheKind::DegeneracyStructure`] cache: content that cache
/// rejected (oversize, a zero budget, or a build that straddled a
/// [`clear`]) gets a fresh identity on every call, so its transformers are
/// rebuilt per call rather than cached. A `DegeneracyStructure` budget of 0
/// therefore also disables completed-transformer caching.
pub fn configure_budgets(budgets: impl IntoIterator<Item = (StructureCacheKind, u64)>) {
    for (kind, bytes) in budgets {
        tenet_core::set_structure_cache_byte_budget(kind, bytes);
    }
}
