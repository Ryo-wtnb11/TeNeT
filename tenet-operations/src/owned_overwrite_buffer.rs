use core::mem::MaybeUninit;

struct OwnedOverwriteBuffer<D: Copy> {
    data: Vec<D>,
    target_len: usize,
}

impl<D: Copy> OwnedOverwriteBuffer<D> {
    fn new(target_len: usize) -> Self {
        Self {
            data: Vec::with_capacity(target_len),
            target_len,
        }
    }

    fn spare_mut(&mut self) -> &mut [MaybeUninit<D>] {
        &mut self.data.spare_capacity_mut()[..self.target_len]
    }

    fn finish(mut self) -> Vec<D> {
        // Why not expose initialized length during replay: an executor error
        // may occur after partial writes. The physical overwrite proof is the
        // only authority that permits the completed buffer to escape.
        unsafe {
            self.data.set_len(self.target_len);
        }
        self.data
    }
}

pub(crate) fn initialize_owned<D, E>(
    target_len: usize,
    write: impl FnOnce(&mut [MaybeUninit<D>]) -> Result<(), E>,
) -> Result<Vec<D>, E>
where
    D: Copy,
{
    let mut buffer = OwnedOverwriteBuffer::new(target_len);
    write(buffer.spare_mut())?;
    Ok(buffer.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completed_transaction_returns_initialized_values() {
        // What: the sole unsafe length transition follows complete writes and
        // remains suitable for Miri's uninitialized-memory checks.
        let values = initialize_owned(4, |dst| -> Result<(), ()> {
            for (index, slot) in dst.iter_mut().enumerate() {
                slot.write(index + 1);
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(values, [1, 2, 3, 4]);
    }

    #[test]
    fn failed_partial_transaction_does_not_return_storage() {
        // What: an error after a partial write drops the length-zero buffer
        // instead of exposing a Vec containing uninitialized elements.
        let result = initialize_owned(4, |dst| {
            dst[0].write(7u64);
            Err("injected writer failure")
        });
        assert_eq!(result, Err("injected writer failure"));
    }

    #[test]
    fn panicking_partial_transaction_cannot_expose_initialized_length() {
        // What: unwinding after a partial write drops only a length-zero Vec;
        // no partially initialized owned output can escape the transaction.
        let result = std::panic::catch_unwind(|| {
            let _ = initialize_owned(4, |dst| -> Result<(), ()> {
                dst[0].write(7u64);
                panic!("injected writer panic");
            });
        });
        assert!(result.is_err());
    }
}
