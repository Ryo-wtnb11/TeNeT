//! Member-strided storage views and the Host replay of one fully-direct
//! coupled-block plan over `B` stacked members (#1287).
//!
//! The member axis exists only here, as a dense stride: the plan's jobs
//! address one member, and a member `i` of an operand sits `i` strides later.
//! No categorical decision is made in this module.

use std::sync::Arc;

use num_traits::{One, Zero};
use tenet_core::{HostReadableStorage, HostWritableStorage, Placement, TensorStorage};
use tenet_dense::strided_batch_runs;

use crate::fusion_replay::{FusionBlockContractPlan, Rank2Gemm, Rank2GemmBatchJob};
use crate::{HostKernelAdapter, OperationError};

fn check_extent(
    storage_len: usize,
    member_len: usize,
    members: usize,
    member_stride: usize,
) -> Result<(), OperationError> {
    if members == 0 {
        return Err(OperationError::InvalidArgument {
            message: "a stacked view needs at least one member",
        });
    }
    if member_stride < member_len {
        return Err(OperationError::InvalidArgument {
            message: "a stacked view's member stride is shorter than a member, so members overlap",
        });
    }
    let end = (members - 1)
        .checked_mul(member_stride)
        .and_then(|start| start.checked_add(member_len))
        .ok_or(OperationError::ElementCountOverflow)?;
    if end > storage_len {
        return Err(OperationError::ElementCountMismatch {
            expected: end,
            actual: storage_len,
        });
    }
    Ok(())
}

/// `B` members of `member_len` elements at `member_stride` inside one
/// storage. Its [`TensorStorage::len`] is one member's length, so a plan's
/// per-job range checks bound one member; only stacked GEMM seams accept it.
#[doc(hidden)]
pub struct StackedStorageView<'a, S> {
    storage: &'a S,
    member_len: usize,
    members: usize,
    member_stride: usize,
}

impl<'a, S> StackedStorageView<'a, S> {
    /// Rejects no members, overlapping members and a last member past the
    /// end of `storage`.
    pub fn new<D>(
        storage: &'a S,
        member_len: usize,
        members: usize,
        member_stride: usize,
    ) -> Result<Self, OperationError>
    where
        S: TensorStorage<D>,
    {
        check_extent(storage.len(), member_len, members, member_stride)?;
        Ok(Self {
            storage,
            member_len,
            members,
            member_stride,
        })
    }

    pub fn storage(&self) -> &'a S {
        self.storage
    }

    pub fn members(&self) -> usize {
        self.members
    }

    pub fn member_stride(&self) -> usize {
        self.member_stride
    }
}

impl<D, S: TensorStorage<D>> TensorStorage<D> for StackedStorageView<'_, S> {
    fn len(&self) -> usize {
        self.member_len
    }

    fn placement(&self) -> Placement {
        self.storage.placement()
    }
}

/// The destination form of [`StackedStorageView`].
#[doc(hidden)]
pub struct StackedStorageViewMut<'a, S> {
    storage: &'a mut S,
    member_len: usize,
    members: usize,
    member_stride: usize,
}

impl<'a, S> StackedStorageViewMut<'a, S> {
    /// The same checks as [`StackedStorageView::new`].
    pub fn new<D>(
        storage: &'a mut S,
        member_len: usize,
        members: usize,
        member_stride: usize,
    ) -> Result<Self, OperationError>
    where
        S: TensorStorage<D>,
    {
        check_extent(storage.len(), member_len, members, member_stride)?;
        Ok(Self {
            storage,
            member_len,
            members,
            member_stride,
        })
    }

    pub fn storage_mut(&mut self) -> &mut S {
        self.storage
    }

    pub fn members(&self) -> usize {
        self.members
    }

    pub fn member_stride(&self) -> usize {
        self.member_stride
    }
}

impl<D, S: TensorStorage<D>> TensorStorage<D> for StackedStorageViewMut<'_, S> {
    fn len(&self) -> usize {
        self.member_len
    }

    fn placement(&self) -> Placement {
        self.storage.placement()
    }
}

/// A fully-direct, identity-orientation, unit-coefficient plan expanded over
/// `B` unpadded members: every plan job repeated at `i · L` of each operand,
/// job-major, so the `B` copies of one job are one affine run.
#[doc(hidden)]
pub struct StackedDirectReplay<C = f64> {
    plan: Arc<FusionBlockContractPlan<C>>,
    members: usize,
    /// `[dst, lhs, rhs]` member lengths, which are also the member strides.
    member_lens: [usize; 3],
    jobs: Vec<Rank2GemmBatchJob>,
    runs: Vec<usize>,
    /// Each inactive destination layout with a trailing member axis.
    inactive: Vec<(Vec<usize>, Vec<isize>, isize)>,
    /// All-zero source strides as long as the longest inactive layout, so a
    /// zero fill borrows a prefix instead of allocating per call.
    zero_strides: Vec<isize>,
}

impl<C> StackedDirectReplay<C>
where
    C: Copy + PartialEq + One,
{
    /// Rejects a plan that is not fully direct, has a transformed operand or
    /// a non-unit job coefficient, or `members == 0`.
    pub fn new(
        plan: Arc<FusionBlockContractPlan<C>>,
        members: usize,
    ) -> Result<Self, OperationError> {
        plan.require_identity_direct_replay()?;
        if members == 0 {
            return Err(OperationError::InvalidArgument {
                message: "a stacked replay needs at least one member",
            });
        }
        let member_lens = plan.member_lens()?;
        let [dst_len, lhs_len, rhs_len] = member_lens;
        let shift = |offset: usize, len: usize, member: usize| {
            member
                .checked_mul(len)
                .and_then(|start| start.checked_add(offset))
                .ok_or(OperationError::ElementCountOverflow)
        };
        let mut jobs = Vec::with_capacity(
            plan.direct_batch()
                .len()
                .checked_mul(members)
                .ok_or(OperationError::ElementCountOverflow)?,
        );
        for job in plan.direct_batch() {
            for member in 0..members {
                jobs.push(Rank2GemmBatchJob {
                    dst_offset: shift(job.dst_offset, dst_len, member)?,
                    lhs_offset: shift(job.lhs_offset, lhs_len, member)?,
                    rhs_offset: shift(job.rhs_offset, rhs_len, member)?,
                    ..*job
                });
            }
        }
        let runs = strided_batch_runs(&jobs);
        let member_stride =
            isize::try_from(dst_len).map_err(|_| OperationError::ElementCountOverflow)?;
        let inactive = plan
            .inactive_destination_regions()
            .iter()
            .map(|layout| {
                let mut shape = layout.block.shape.clone();
                shape.push(members);
                let mut strides = layout.block.strides.clone();
                strides.push(member_stride);
                (shape, strides, layout.block.offset)
            })
            .collect::<Vec<_>>();
        let zero_strides = vec![
            0;
            inactive
                .iter()
                .map(|(shape, _, _)| shape.len())
                .max()
                .unwrap_or(0)
        ];
        Ok(Self {
            plan,
            members,
            member_lens,
            jobs,
            runs,
            inactive,
            zero_strides,
        })
    }

    pub fn plan(&self) -> &Arc<FusionBlockContractPlan<C>> {
        &self.plan
    }

    pub fn members(&self) -> usize {
        self.members
    }

    /// Host bytes this replay holds beyond the shared plan.
    pub fn retained_bytes(&self) -> usize {
        self.jobs.capacity() * std::mem::size_of::<Rank2GemmBatchJob>()
            + self.runs.capacity() * std::mem::size_of::<usize>()
            + self.zero_strides.capacity() * std::mem::size_of::<isize>()
            + self
                .inactive
                .iter()
                .map(|(shape, strides, _)| {
                    shape.capacity() * std::mem::size_of::<usize>()
                        + strides.capacity() * std::mem::size_of::<isize>()
                })
                .sum::<usize>()
    }

    /// `dst[i] = lhs[i] · rhs[i]` for every member, as one batch submission.
    ///
    /// With `zero_inactive`, each inactive destination layout is first
    /// zero-filled over all members in one strided fill, so the result does
    /// not depend on what `dst` held; without it the caller guarantees those
    /// layouts are already zero. Active blocks are overwritten (`beta = 0`).
    #[allow(clippy::too_many_arguments)]
    pub fn execute_host<A, G, D, SD, SL, SR>(
        &self,
        kernels: &mut A,
        gemm: &mut G,
        dst: &mut StackedStorageViewMut<'_, SD>,
        lhs: &StackedStorageView<'_, SL>,
        rhs: &StackedStorageView<'_, SR>,
        zero_inactive: bool,
    ) -> Result<(), OperationError>
    where
        A: HostKernelAdapter<D>,
        G: Rank2Gemm<D>,
        D: Copy + Zero + One,
        SD: HostWritableStorage<D>,
        SL: HostReadableStorage<D>,
        SR: HostReadableStorage<D>,
    {
        let [dst_len, lhs_len, rhs_len] = self.member_lens;
        for (members, len, stride, expected) in [
            (dst.members, dst.member_len, dst.member_stride, dst_len),
            (lhs.members, lhs.member_len, lhs.member_stride, lhs_len),
            (rhs.members, rhs.member_len, rhs.member_stride, rhs_len),
        ] {
            if members != self.members || len != expected || stride != expected {
                return Err(OperationError::InvalidArgument {
                    message: "stacked operand does not match the replay's members or layout",
                });
            }
        }
        let dst_data = dst.storage.as_mut_slice();
        if zero_inactive {
            let zero = [D::zero()];
            for (shape, strides, offset) in &self.inactive {
                kernels.copy_scale_strided(
                    dst_data,
                    &zero,
                    shape,
                    strides,
                    &self.zero_strides[..shape.len()],
                    *offset,
                    0,
                    false,
                    D::one(),
                )?;
            }
        }
        gemm.matmul_rank2_batch(
            dst_data,
            lhs.storage.as_slice(),
            rhs.storage.as_slice(),
            &self.jobs,
            &self.runs,
            D::one(),
            D::zero(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn views_reject_overlap_and_out_of_bounds_members() {
        let storage = vec![0.0_f64; 10];
        assert!(StackedStorageView::new::<f64>(&storage, 3, 3, 3).is_ok());
        assert!(StackedStorageView::new::<f64>(&storage, 3, 3, 4).is_err());
        assert!(StackedStorageView::new::<f64>(&storage, 3, 2, 2).is_err());
        assert!(StackedStorageView::new::<f64>(&storage, 3, 0, 3).is_err());
        let view = StackedStorageView::new::<f64>(&storage, 3, 2, 7).unwrap();
        assert_eq!(TensorStorage::<f64>::len(&view), 3);
        let mut storage = storage;
        assert!(StackedStorageViewMut::new::<f64>(&mut storage, 4, 3, 4).is_err());
    }
}
