//! Storage views over a member-stacked payload (#1287).
//!
//! A batched handle executes one per-member plan over `members` payloads laid
//! out in one buffer, member `i` at `i * member_stride`. The view reports the
//! length of **one** member through [`TensorStorage::len`], so a plan's
//! per-job range validation checks every job against one member; only a
//! batched GEMM seam that walks the member axis may accept these views.

use tenet_core::{Placement, TensorStorage};

use crate::OperationError;

/// Checks that `members` members of `member_len` elements at `member_stride`
/// neither overlap nor read past `storage_len`.
fn validate(
    storage_len: usize,
    member_len: usize,
    members: usize,
    member_stride: usize,
) -> Result<(), OperationError> {
    if member_stride < member_len {
        return Err(OperationError::InvalidArgument {
            message: "stacked member stride is smaller than the member length",
        });
    }
    let Some(last) = members.checked_sub(1) else {
        return Ok(());
    };
    let end = last
        .checked_mul(member_stride)
        .and_then(|offset| offset.checked_add(member_len))
        .ok_or(OperationError::ElementCountOverflow)?;
    if end > storage_len {
        return Err(OperationError::ElementCountMismatch {
            expected: end,
            actual: storage_len,
        });
    }
    Ok(())
}

/// Read view of `members` payloads of `member_len` elements, member `i` at
/// `i * member_stride` of `storage`.
#[doc(hidden)]
#[derive(Debug)]
pub struct StackedStorageView<'a, S> {
    storage: &'a S,
    member_len: usize,
    members: usize,
    member_stride: usize,
}

impl<'a, S> StackedStorageView<'a, S> {
    /// Rejects `member_stride < member_len` and any member past the buffer.
    pub fn new<D>(
        storage: &'a S,
        member_len: usize,
        members: usize,
        member_stride: usize,
    ) -> Result<Self, OperationError>
    where
        S: TensorStorage<D>,
    {
        validate(storage.len(), member_len, members, member_stride)?;
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

/// Destination form of [`StackedStorageView`].
#[doc(hidden)]
#[derive(Debug)]
pub struct StackedStorageViewMut<'a, S> {
    storage: &'a mut S,
    member_len: usize,
    members: usize,
    member_stride: usize,
}

impl<'a, S> StackedStorageViewMut<'a, S> {
    /// Rejects `member_stride < member_len` and any member past the buffer.
    pub fn new<D>(
        storage: &'a mut S,
        member_len: usize,
        members: usize,
        member_stride: usize,
    ) -> Result<Self, OperationError>
    where
        S: TensorStorage<D>,
    {
        validate(storage.len(), member_len, members, member_stride)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn view(
        storage: &Vec<f64>,
        len: usize,
        members: usize,
        stride: usize,
    ) -> Result<usize, OperationError> {
        StackedStorageView::new::<f64>(storage, len, members, stride)
            .map(|view| TensorStorage::<f64>::len(&view))
    }

    fn view_mut(
        storage: &mut Vec<f64>,
        len: usize,
        members: usize,
        stride: usize,
    ) -> Result<usize, OperationError> {
        StackedStorageViewMut::new::<f64>(storage, len, members, stride)
            .map(|view| TensorStorage::<f64>::len(&view))
    }

    #[test]
    fn constructors_check_stride_and_bounds() {
        // What: 3 members of 4 at stride 5 end at 14; stride < len, one
        // element short, and index overflow are rejected identically by the
        // read and the destination form, and `len` is one member.
        let mut storage = vec![0.0; 14];
        for check in [
            |s: &mut Vec<f64>, l, m, t| view(s, l, m, t),
            |s: &mut Vec<f64>, l, m, t| view_mut(s, l, m, t),
        ] {
            assert_eq!(check(&mut storage, 4, 3, 5), Ok(4));
            assert_eq!(check(&mut storage, 4, 3, 4), Ok(4));
            assert_eq!(check(&mut storage, 0, 0, 0), Ok(0));
            assert!(matches!(
                check(&mut storage, 4, 3, 3),
                Err(OperationError::InvalidArgument { .. })
            ));
            assert_eq!(
                check(&mut storage, 5, 3, 5),
                Err(OperationError::ElementCountMismatch {
                    expected: 15,
                    actual: 14
                })
            );
            assert_eq!(
                check(&mut storage, 1, usize::MAX, 2),
                Err(OperationError::ElementCountOverflow)
            );
        }
    }
}
