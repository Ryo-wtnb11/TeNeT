/// Host-owned scratch buffer used by current raw host-slice replay paths.
///
/// This keeps direct `Vec<T>` ownership out of categorical replay workspaces.
/// Future storage-aware workspaces can replace this boundary with scratch
/// buffers allocated from `TensorMap::similar_storage_filled`.
#[derive(Clone, Debug)]
pub(crate) struct HostScratchBuffer<T> {
    data: Vec<T>,
}

impl<T> Default for HostScratchBuffer<T> {
    fn default() -> Self {
        Self { data: Vec::new() }
    }
}

impl<T> HostScratchBuffer<T> {
    #[inline]
    pub(crate) fn filled(len: usize, value: T) -> Self
    where
        T: Clone,
    {
        Self {
            data: vec![value; len],
        }
    }

    #[inline]
    pub(crate) fn resize_filled(&mut self, len: usize, value: T)
    where
        T: Clone,
    {
        self.data.resize(len, value);
    }

    #[inline]
    pub(crate) fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        self.data.fill(value);
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub(crate) fn as_slice(&self) -> &[T] {
        &self.data
    }

    #[inline]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}
