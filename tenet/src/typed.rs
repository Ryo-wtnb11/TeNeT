//! Provider-typed facade: spaces and tensor maps that keep the concrete
//! fusion-rule type `R` and speak the provider's own sector labels.
//!
//! The ergonomic [`crate::prelude`] facade erases the rule behind a fixed set
//! of built-ins. This module is its typed sibling: `R` stays concrete through
//! monomorphized construction, so any provider — including one defined
//! downstream — can drive it, and inspection returns
//! [`SectorCodec::Sector`] labels instead of opaque
//! [`tenet_core::SectorId`] keys. The engine itself never sees a label; the
//! codec is the single boundary where one enters or leaves.
//!
//! # Phase boundary
//!
//! This is the phase-2 surface of issue #557: **construction and inspection
//! only**. A typed tensor map can be built and read (spaces, blocks, labels,
//! data), and nothing else. Transforms and contractions are phase 3 and are
//! deliberately not reachable from here yet, so the operation surface gets its
//! own review before it exists. The module is not in the prelude for the same
//! reason.
//!
//! Construction consumes only the transactional checked admission path, so a
//! provider that reports an invalid or unrepresentable algebra fails with a
//! typed error and publishes no layout, cache, or admission state.

use std::sync::Arc;

use tenet_core::{CheckedFusionAlgebra, SectorLeg};

pub use tenet_core::SectorCodec;

use crate::error::Error;

/// One tensor leg: a provider plus the sector-to-degeneracy map of that axis
/// (TensorKit's `GradedSpace`).
///
/// The leg owns the complete map independently of which fusion trees a tensor
/// built on it happens to populate. `is_dual` marks the conjugate space
/// (TensorKit's `V'`).
pub struct GradedSpace<R> {
    provider: Arc<R>,
    leg: SectorLeg,
}

// Why hand-written instead of derived: the derives would demand `R: Clone` and
// `R: Debug`, neither of which a provider needs to satisfy — the provider is
// shared behind an `Arc` and its labels, not the rule itself, are what a
// diagnostic wants to show.
impl<R> Clone for GradedSpace<R> {
    fn clone(&self) -> Self {
        Self {
            provider: Arc::clone(&self.provider),
            leg: self.leg.clone(),
        }
    }
}

impl<R> core::fmt::Debug for GradedSpace<R> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("GradedSpace")
            .field("leg", &self.leg)
            .finish_non_exhaustive()
    }
}

impl<R> GradedSpace<R>
where
    R: SectorCodec + CheckedFusionAlgebra,
{
    /// Builds a leg from `(label, degeneracy)` pairs.
    ///
    /// Order is irrelevant: the leg stores its sectors in the provider's
    /// [`tenet_core::SectorId`] order. A zero-degeneracy sector is absent from
    /// the result, matching the leg invariant of the erased facade.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] when a label is declared more than once,
    ///   naming the offending label.
    /// - [`Error::InvalidArgument`] when two distinct labels encode to one
    ///   sector id. That is the provider breaking [`SectorCodec`]'s
    ///   injectivity law, and it is reported as such rather than as a caller
    ///   duplicate.
    /// - [`Error::FusionAlgebra`] when the provider cannot represent a label,
    ///   preserving the provider's own encode error.
    pub fn try_new<Pairs>(provider: Arc<R>, pairs: Pairs, is_dual: bool) -> Result<Self, Error>
    where
        Pairs: IntoIterator<Item = (R::Sector, usize)>,
    {
        let pairs: Vec<(R::Sector, usize)> = pairs.into_iter().collect();
        // Why duplicate detection precedes `SectorLeg::try_new`: the leg only
        // ever sees encoded ids, so its own duplicate error can only name a
        // `SectorId`. Diagnosing here lets the caller see the label it wrote,
        // and separates a caller duplicate from a provider whose codec aliases
        // two labels onto one id — the leg cannot tell those apart at all.
        let mut sorted: Vec<&R::Sector> = pairs.iter().map(|(label, _)| label).collect();
        sorted.sort_unstable();
        if let Some(window) = sorted.windows(2).find(|window| window[0] == window[1]) {
            return Err(Error::InvalidArgument(format!(
                "sector label {:?} is declared more than once",
                window[0]
            )));
        }

        let mut encoded = Vec::with_capacity(pairs.len());
        for (label, degeneracy) in &pairs {
            encoded.push((provider.encode_sector(label)?, label, *degeneracy));
        }
        let mut by_id: Vec<_> = encoded.iter().collect();
        by_id.sort_unstable_by_key(|(id, _, _)| *id);
        if let Some(window) = by_id.windows(2).find(|window| window[0].0 == window[1].0) {
            return Err(Error::InvalidArgument(format!(
                "SectorCodec law violation: labels {:?} and {:?} both encode to {:?}",
                window[0].1, window[1].1, window[0].0
            )));
        }

        let leg = SectorLeg::try_new(
            encoded.iter().map(|(id, _, degeneracy)| (*id, *degeneracy)),
            is_dual,
        )
        .map_err(|error| Error::InvalidArgument(error.to_string()))?;
        Ok(Self { provider, leg })
    }

    /// The sector labels carried by this leg, in the provider's
    /// [`tenet_core::SectorId`] order.
    ///
    /// The order is the engine's, deliberately: it is the order of
    /// [`Self::degeneracies`] and of every block layout derived from the leg,
    /// so re-sorting by label here would desynchronize the two. A caller that
    /// wants label order sorts the returned vector.
    ///
    /// # Errors
    ///
    /// Returns [`Error::FusionAlgebra`] when the provider cannot decode an id
    /// it previously produced, which is a violation of [`SectorCodec`]'s
    /// decode-totality law.
    pub fn sectors(&self) -> Result<Vec<R::Sector>, Error> {
        self.leg
            .sectors()
            .iter()
            .map(|&id| self.provider.decode_sector(id).map_err(Error::from))
            .collect()
    }

    /// Per-sector degeneracies, parallel to [`Self::sectors`].
    #[inline]
    pub fn degeneracies(&self) -> &[usize] {
        self.leg.degeneracies()
    }

    /// Whether this is the conjugate space (TensorKit's `V'`).
    #[inline]
    pub fn is_dual(&self) -> bool {
        self.leg.is_dual()
    }

    /// The provider this leg is bound to.
    #[inline]
    pub fn provider(&self) -> &R {
        &self.provider
    }

    /// The conjugate leg: every sector replaced by its dual (degeneracies
    /// carried along) and the dual flag flipped.
    ///
    /// # Errors
    ///
    /// Returns [`Error::FusionAlgebra`] when the provider cannot represent the
    /// dual of some sector. No partially dualized leg is produced.
    pub fn try_dual(&self) -> Result<Self, Error> {
        Ok(Self {
            provider: Arc::clone(&self.provider),
            leg: self.leg.try_dual(self.provider.as_ref())?,
        })
    }
}
