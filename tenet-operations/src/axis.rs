use crate::error::OperationError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AxisPermutation<'a> {
    Identity,
    Axes(&'a [usize]),
}

impl<'a> AxisPermutation<'a> {
    #[inline]
    pub fn identity() -> Self {
        Self::Identity
    }

    #[inline]
    pub fn from_axes(axes: &'a [usize]) -> Self {
        Self::Axes(axes)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorContractAxisSpec<'a> {
    lhs_contracting_axes: &'a [usize],
    rhs_contracting_axes: &'a [usize],
    output_permutation: AxisPermutation<'a>,
}

impl<'a> TensorContractAxisSpec<'a> {
    pub fn new(
        lhs_contracting_axes: &'a [usize],
        rhs_contracting_axes: &'a [usize],
        output_permutation: AxisPermutation<'a>,
    ) -> Self {
        Self {
            lhs_contracting_axes,
            rhs_contracting_axes,
            output_permutation,
        }
    }

    pub fn canonical(lhs_contracting_axes: &'a [usize], rhs_contracting_axes: &'a [usize]) -> Self {
        Self::new(
            lhs_contracting_axes,
            rhs_contracting_axes,
            AxisPermutation::identity(),
        )
    }

    #[inline]
    pub fn lhs_contracting_axes(&self) -> &'a [usize] {
        self.lhs_contracting_axes
    }

    #[inline]
    pub fn rhs_contracting_axes(&self) -> &'a [usize] {
        self.rhs_contracting_axes
    }

    #[inline]
    pub fn output_permutation(&self) -> AxisPermutation<'a> {
        self.output_permutation
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct OwnedTensorContractAxisSpec {
    lhs_contracting_axes: Vec<usize>,
    rhs_contracting_axes: Vec<usize>,
    output_axes: Vec<usize>,
}

impl OwnedTensorContractAxisSpec {
    pub fn new(
        lhs_contracting_axes: Vec<usize>,
        rhs_contracting_axes: Vec<usize>,
        output_axes: Vec<usize>,
    ) -> Self {
        Self {
            lhs_contracting_axes,
            rhs_contracting_axes,
            output_axes,
        }
    }

    #[inline]
    pub fn as_spec(&self) -> TensorContractAxisSpec<'_> {
        TensorContractAxisSpec::new(
            self.lhs_contracting_axes.as_slice(),
            self.rhs_contracting_axes.as_slice(),
            AxisPermutation::from_axes(self.output_axes.as_slice()),
        )
    }

    #[inline]
    pub fn lhs_contracting_axes(&self) -> &[usize] {
        self.lhs_contracting_axes.as_slice()
    }

    #[inline]
    pub fn rhs_contracting_axes(&self) -> &[usize] {
        self.rhs_contracting_axes.as_slice()
    }

    #[inline]
    pub fn output_axes(&self) -> &[usize] {
        self.output_axes.as_slice()
    }
}

pub(crate) fn permutation_axes(
    permutation: AxisPermutation<'_>,
    rank: usize,
) -> Result<Vec<usize>, OperationError> {
    match permutation {
        AxisPermutation::Identity => Ok((0..rank).collect()),
        AxisPermutation::Axes(axes) => {
            if axes.len() != rank {
                return Err(OperationError::InvalidPermutation {
                    axes: axes.to_vec(),
                    rank,
                });
            }
            let mut seen = vec![false; rank];
            for &axis in axes {
                if axis >= rank || seen[axis] {
                    return Err(OperationError::InvalidPermutation {
                        axes: axes.to_vec(),
                        rank,
                    });
                }
                seen[axis] = true;
            }
            Ok(axes.to_vec())
        }
    }
}
