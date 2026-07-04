use crate::OperationError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutputAxisOrder<'a> {
    Identity,
    Axes(&'a [usize]),
}

impl<'a> OutputAxisOrder<'a> {
    #[inline]
    pub fn identity() -> Self {
        Self::Identity
    }

    #[inline]
    pub fn from_axes(axes: &'a [usize]) -> Self {
        Self::Axes(axes)
    }
}

/// Full index lowering for a pairwise tensor contraction.
///
/// TensorKit / TensorOperations.jl correspondence:
/// `pA = (open axes of lhs, contracted axes of lhs)`,
/// `pB = (contracted axes of rhs, open axes of rhs)`,
/// `pAB = output axis order` (here [`OutputAxisOrder`]), plus the
/// `conjA` / `conjB` conjugation flags (`lhs_conjugate` / `rhs_conjugate`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorContractSpec<'a> {
    lhs_contracting_axes: &'a [usize],
    rhs_contracting_axes: &'a [usize],
    output_permutation: OutputAxisOrder<'a>,
    lhs_conjugate: bool,
    rhs_conjugate: bool,
}

impl<'a> TensorContractSpec<'a> {
    pub fn new(
        lhs_contracting_axes: &'a [usize],
        rhs_contracting_axes: &'a [usize],
        output_permutation: OutputAxisOrder<'a>,
    ) -> Self {
        Self::new_with_conjugation(
            lhs_contracting_axes,
            rhs_contracting_axes,
            output_permutation,
            false,
            false,
        )
    }

    pub fn new_with_conjugation(
        lhs_contracting_axes: &'a [usize],
        rhs_contracting_axes: &'a [usize],
        output_permutation: OutputAxisOrder<'a>,
        lhs_conjugate: bool,
        rhs_conjugate: bool,
    ) -> Self {
        Self {
            lhs_contracting_axes,
            rhs_contracting_axes,
            output_permutation,
            lhs_conjugate,
            rhs_conjugate,
        }
    }

    /// Contract the given axes with the default output order (`pAB` omitted):
    /// lhs open axes in original order, then rhs open axes in original order.
    pub fn with_default_output_order(
        lhs_contracting_axes: &'a [usize],
        rhs_contracting_axes: &'a [usize],
    ) -> Self {
        Self::new(
            lhs_contracting_axes,
            rhs_contracting_axes,
            OutputAxisOrder::identity(),
        )
    }

    pub fn with_default_output_order_and_conjugation(
        lhs_contracting_axes: &'a [usize],
        rhs_contracting_axes: &'a [usize],
        lhs_conjugate: bool,
        rhs_conjugate: bool,
    ) -> Self {
        Self::new_with_conjugation(
            lhs_contracting_axes,
            rhs_contracting_axes,
            OutputAxisOrder::identity(),
            lhs_conjugate,
            rhs_conjugate,
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
    pub fn output_permutation(&self) -> OutputAxisOrder<'a> {
        self.output_permutation
    }

    #[inline]
    pub fn lhs_conjugate(&self) -> bool {
        self.lhs_conjugate
    }

    #[inline]
    pub fn rhs_conjugate(&self) -> bool {
        self.rhs_conjugate
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct TensorContractSpecOwned {
    lhs_contracting_axes: Vec<usize>,
    rhs_contracting_axes: Vec<usize>,
    output_axes: Vec<usize>,
    lhs_conjugate: bool,
    rhs_conjugate: bool,
}

impl TensorContractSpecOwned {
    pub fn new(
        lhs_contracting_axes: Vec<usize>,
        rhs_contracting_axes: Vec<usize>,
        output_axes: Vec<usize>,
    ) -> Self {
        Self::new_with_conjugation(
            lhs_contracting_axes,
            rhs_contracting_axes,
            output_axes,
            false,
            false,
        )
    }

    pub fn new_with_conjugation(
        lhs_contracting_axes: Vec<usize>,
        rhs_contracting_axes: Vec<usize>,
        output_axes: Vec<usize>,
        lhs_conjugate: bool,
        rhs_conjugate: bool,
    ) -> Self {
        Self {
            lhs_contracting_axes,
            rhs_contracting_axes,
            output_axes,
            lhs_conjugate,
            rhs_conjugate,
        }
    }

    #[inline]
    pub fn as_spec(&self) -> TensorContractSpec<'_> {
        TensorContractSpec::new_with_conjugation(
            self.lhs_contracting_axes.as_slice(),
            self.rhs_contracting_axes.as_slice(),
            OutputAxisOrder::from_axes(self.output_axes.as_slice()),
            self.lhs_conjugate,
            self.rhs_conjugate,
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

    #[inline]
    pub fn lhs_conjugate(&self) -> bool {
        self.lhs_conjugate
    }

    #[inline]
    pub fn rhs_conjugate(&self) -> bool {
        self.rhs_conjugate
    }
}

pub fn permutation_axes(
    permutation: OutputAxisOrder<'_>,
    rank: usize,
) -> Result<Vec<usize>, OperationError> {
    match permutation {
        OutputAxisOrder::Identity => Ok((0..rank).collect()),
        OutputAxisOrder::Axes(axes) => {
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorTraceAxisSpec<'a> {
    output_axes: &'a [usize],
    trace_lhs_axes: &'a [usize],
    trace_rhs_axes: &'a [usize],
    source_conjugate: bool,
}

impl<'a> TensorTraceAxisSpec<'a> {
    pub fn new(
        output_axes: &'a [usize],
        trace_lhs_axes: &'a [usize],
        trace_rhs_axes: &'a [usize],
    ) -> Self {
        Self::new_with_conjugation(output_axes, trace_lhs_axes, trace_rhs_axes, false)
    }

    pub fn new_with_conjugation(
        output_axes: &'a [usize],
        trace_lhs_axes: &'a [usize],
        trace_rhs_axes: &'a [usize],
        source_conjugate: bool,
    ) -> Self {
        Self {
            output_axes,
            trace_lhs_axes,
            trace_rhs_axes,
            source_conjugate,
        }
    }

    #[inline]
    pub fn output_axes(&self) -> &'a [usize] {
        self.output_axes
    }

    #[inline]
    pub fn trace_lhs_axes(&self) -> &'a [usize] {
        self.trace_lhs_axes
    }

    #[inline]
    pub fn trace_rhs_axes(&self) -> &'a [usize] {
        self.trace_rhs_axes
    }

    #[inline]
    pub fn source_conjugate(&self) -> bool {
        self.source_conjugate
    }
}
