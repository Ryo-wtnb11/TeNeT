//! Test probes of tenet-network (#2221): one thread-local ledger of
//! [`Event`] counts, read through [`record`].
//!
//! All three events fire on the calling thread (steps and slice jobs run
//! sequentially there), so a concurrent test cannot perturb the count
//! (#1568).

use std::cell::Cell;

/// One observed occurrence in network execution. Each fact has one event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Event {
    /// A symmetric sliced execution finished one partial contraction.
    SliceJobCompleted,
    /// The payload meter snapshotted the intermediate buffers.
    IntermediatePayloadSnapshot,
    /// A network step contracted two operands.
    ContractStep,
}

const EVENTS: usize = Event::ContractStep as usize + 1;

std::thread_local! {
    static COUNTS: Cell<[usize; EVENTS]> = const { Cell::new([0; EVENTS]) };
}

/// Records one `event`.
pub(crate) fn hit(event: Event) {
    COUNTS.with(|counts| {
        let mut next = counts.get();
        next[event as usize] += 1;
        counts.set(next);
    });
}

/// What this thread recorded since [`record`] returned this guard.
pub(crate) struct Recording([usize; EVENTS]);

/// Starts a recording.
pub(crate) fn record() -> Recording {
    Recording(COUNTS.with(Cell::get))
}

impl Recording {
    /// Occurrences of `event` since the recording started.
    pub(crate) fn get(&self, event: Event) -> usize {
        COUNTS.with(Cell::get)[event as usize] - self.0[event as usize]
    }
}
