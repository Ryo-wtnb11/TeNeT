use tenet_core::Placement;

/// Reports the storage/execution placement represented by an execution object.
///
/// This is an inspection boundary, not a dispatch contract. Host-slice replay
/// backends and workspaces report `Placement::Host`; future device/MPI
/// execution should add separate placement-aware execution traits instead of
/// implementing the host-slice APIs.
pub trait ReportsPlacement {
    fn placement(&self) -> Placement;

    #[inline]
    fn is_host_placement(&self) -> bool {
        self.placement() == Placement::Host
    }
}
