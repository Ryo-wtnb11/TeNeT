use core::convert::Infallible;
use core::mem::MaybeUninit;
use core::ops::Range;

use num_complex::Complex64;

use crate::owned_overwrite_buffer::initialize_owned;
use crate::ConjugateValue;

#[cfg(test)]
thread_local! {
    static TEST_WRITE_BITMAP: std::cell::RefCell<Option<Vec<usize>>> =
        const { std::cell::RefCell::new(None) };
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OwnedCatSide {
    Domain,
    Codomain,
}

/// Physical copy geometry compiled by the categorical concatenation authority.
#[doc(hidden)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OwnedCatCopy {
    source: usize,
    source_offset: usize,
    destination_offset: usize,
    rows: usize,
    cols: usize,
    source_row_stride: usize,
    source_column_stride: usize,
    destination_leading_dimension: usize,
    destination_region_start: usize,
    destination_region_end: usize,
    conjugate: bool,
}

impl OwnedCatCopy {
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        source: usize,
        source_offset: usize,
        destination_offset: usize,
        shape: [usize; 2],
        source_strides: [usize; 2],
        destination_leading_dimension: usize,
        destination_region: Range<usize>,
        conjugate: bool,
    ) -> Self {
        Self {
            source,
            source_offset,
            destination_offset,
            rows: shape[0],
            cols: shape[1],
            source_row_stride: source_strides[0],
            source_column_stride: source_strides[1],
            destination_leading_dimension,
            destination_region_start: destination_region.start,
            destination_region_end: destination_region.end,
            conjugate,
        }
    }

    #[doc(hidden)]
    pub fn source(&self) -> usize {
        self.source
    }

    #[doc(hidden)]
    pub fn source_offset(&self) -> usize {
        self.source_offset
    }

    #[doc(hidden)]
    pub fn destination_offset(&self) -> usize {
        self.destination_offset
    }

    #[doc(hidden)]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[doc(hidden)]
    pub fn cols(&self) -> usize {
        self.cols
    }

    #[doc(hidden)]
    pub fn source_row_stride(&self) -> usize {
        self.source_row_stride
    }

    #[doc(hidden)]
    pub fn source_column_stride(&self) -> usize {
        self.source_column_stride
    }

    #[doc(hidden)]
    pub fn destination_leading_dimension(&self) -> usize {
        self.destination_leading_dimension
    }

    #[doc(hidden)]
    pub fn conjugate(&self) -> bool {
        self.conjugate
    }
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub enum OwnedCatC64Source<'a> {
    F64(&'a [f64]),
    C64(&'a [Complex64]),
}

/// Returns `None` unless the compiled copies prove a complete physical overwrite.
#[doc(hidden)]
pub fn try_cat_owned_raw<D: ConjugateValue>(
    required_len: usize,
    side: OwnedCatSide,
    copies: &[OwnedCatCopy],
    sources: [&[D]; 2],
) -> Option<Vec<D>> {
    validate_owned_cat(
        required_len,
        side,
        copies,
        [sources[0].len(), sources[1].len()],
    )?;
    Some(initialize_infallible(required_len, |destination| {
        for copy in copies {
            write_same(destination, sources[copy.source], copy);
        }
    }))
}

/// Returns `None` unless the compiled mixed-dtype copies prove a complete overwrite.
#[doc(hidden)]
pub fn try_cat_owned_c64_raw(
    required_len: usize,
    side: OwnedCatSide,
    copies: &[OwnedCatCopy],
    sources: [OwnedCatC64Source<'_>; 2],
) -> Option<Vec<Complex64>> {
    let source_lengths = sources.map(|source| match source {
        OwnedCatC64Source::F64(values) => values.len(),
        OwnedCatC64Source::C64(values) => values.len(),
    });
    validate_owned_cat(required_len, side, copies, source_lengths)?;
    Some(initialize_infallible(required_len, |destination| {
        for copy in copies {
            match sources[copy.source] {
                OwnedCatC64Source::F64(source) => write_widened(destination, source, copy),
                OwnedCatC64Source::C64(source) => write_same(destination, source, copy),
            }
        }
    }))
}

fn initialize_infallible<D: Copy>(
    required_len: usize,
    write: impl FnOnce(&mut [MaybeUninit<D>]),
) -> Vec<D> {
    match initialize_owned(required_len, |destination| -> Result<(), Infallible> {
        write(destination);
        Ok(())
    }) {
        Ok(values) => values,
        Err(error) => match error {},
    }
}

fn validate_owned_cat(
    required_len: usize,
    side: OwnedCatSide,
    copies: &[OwnedCatCopy],
    source_lengths: [usize; 2],
) -> Option<()> {
    if copies.is_empty() {
        return (required_len == 0).then_some(());
    }

    let mut covered_end = 0usize;
    let mut copy_index = 0usize;
    while copy_index < copies.len() {
        let first = &copies[copy_index];
        let region_start = first.destination_region_start;
        let region_end = first.destination_region_end;
        let leading_dimension = first.destination_leading_dimension;
        if region_start != covered_end || region_end < region_start || region_end > required_len {
            return None;
        }
        let region_len = region_end.checked_sub(region_start)?;
        if leading_dimension == 0 {
            if region_len != 0 {
                return None;
            }
        } else if region_len % leading_dimension != 0 {
            return None;
        }
        let region_cols = region_len
            .checked_div(leading_dimension)
            .unwrap_or_default();
        let mut slab_extent = 0usize;
        let mut previous_source = None;

        while copy_index < copies.len()
            && copies[copy_index].destination_region_start == region_start
            && copies[copy_index].destination_region_end == region_end
        {
            let copy = &copies[copy_index];
            if copy.source >= source_lengths.len()
                || copy.destination_leading_dimension != leading_dimension
                || previous_source.is_some_and(|source| copy.source <= source)
                || !source_bounds_hold(copy, source_lengths[copy.source])
            {
                return None;
            }
            previous_source = Some(copy.source);

            match side {
                OwnedCatSide::Domain => {
                    if copy.rows != leading_dimension
                        || copy.destination_offset != region_start.checked_add(slab_extent)?
                    {
                        return None;
                    }
                    slab_extent = slab_extent.checked_add(copy.rows.checked_mul(copy.cols)?)?;
                }
                OwnedCatSide::Codomain => {
                    if copy.cols != region_cols
                        || copy.destination_offset != region_start.checked_add(slab_extent)?
                    {
                        return None;
                    }
                    slab_extent = slab_extent.checked_add(copy.rows)?;
                }
            }
            copy_index += 1;
        }

        let expected_extent = match side {
            OwnedCatSide::Domain => region_len,
            OwnedCatSide::Codomain => leading_dimension,
        };
        if slab_extent != expected_extent {
            return None;
        }
        covered_end = region_end;
    }

    (covered_end == required_len).then_some(())
}

fn source_bounds_hold(copy: &OwnedCatCopy, source_len: usize) -> bool {
    if copy.rows == 0 || copy.cols == 0 {
        return copy.source_offset <= source_len;
    }
    let Some(last_row) = (copy.rows - 1).checked_mul(copy.source_row_stride) else {
        return false;
    };
    let Some(last_column) = (copy.cols - 1).checked_mul(copy.source_column_stride) else {
        return false;
    };
    copy.source_offset
        .checked_add(last_row)
        .and_then(|offset| offset.checked_add(last_column))
        .is_some_and(|last| last < source_len)
}

fn write_same<D: ConjugateValue>(
    destination: &mut [MaybeUninit<D>],
    source: &[D],
    copy: &OwnedCatCopy,
) {
    if copy.rows == 0 || copy.cols == 0 {
        return;
    }
    if !copy.conjugate
        && copy.source_row_stride == 1
        && copy.source_column_stride == copy.rows
        && copy.destination_leading_dimension == copy.rows
    {
        let element_count = copy.rows * copy.cols;
        #[cfg(test)]
        observe_writes(copy.destination_offset..copy.destination_offset + element_count);
        destination[copy.destination_offset..copy.destination_offset + element_count]
            .write_copy_of_slice(&source[copy.source_offset..copy.source_offset + element_count]);
        return;
    }
    for column in 0..copy.cols {
        let source_start = copy.source_offset + column * copy.source_column_stride;
        let destination_start =
            copy.destination_offset + column * copy.destination_leading_dimension;
        if !copy.conjugate && copy.source_row_stride == 1 {
            #[cfg(test)]
            observe_writes(destination_start..destination_start + copy.rows);
            destination[destination_start..destination_start + copy.rows]
                .write_copy_of_slice(&source[source_start..source_start + copy.rows]);
        } else {
            for row in 0..copy.rows {
                #[cfg(test)]
                observe_write(destination_start + row);
                destination[destination_start + row].write(
                    source[source_start + row * copy.source_row_stride].maybe_conj(copy.conjugate),
                );
            }
        }
    }
}

fn write_widened(destination: &mut [MaybeUninit<Complex64>], source: &[f64], copy: &OwnedCatCopy) {
    if copy.rows == 0 || copy.cols == 0 {
        return;
    }
    for column in 0..copy.cols {
        let source_start = copy.source_offset + column * copy.source_column_stride;
        let destination_start =
            copy.destination_offset + column * copy.destination_leading_dimension;
        for row in 0..copy.rows {
            #[cfg(test)]
            observe_write(destination_start + row);
            destination[destination_start + row].write(Complex64::new(
                source[source_start + row * copy.source_row_stride],
                0.0,
            ));
        }
    }
}

#[cfg(test)]
fn observe_write(index: usize) {
    TEST_WRITE_BITMAP.with(|bitmap| {
        if let Some(bitmap) = bitmap.borrow_mut().as_mut() {
            bitmap[index] += 1;
        }
    });
}

#[cfg(test)]
fn observe_writes(range: Range<usize>) {
    for index in range {
        observe_write(index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    fn copy(
        source: usize,
        source_offset: usize,
        destination_offset: usize,
        shape: [usize; 2],
        source_strides: [usize; 2],
        leading_dimension: usize,
        region: Range<usize>,
        conjugate: bool,
    ) -> OwnedCatCopy {
        OwnedCatCopy::new(
            source,
            source_offset,
            destination_offset,
            shape,
            source_strides,
            leading_dimension,
            region,
            conjugate,
        )
    }

    fn with_bitmap<D>(len: usize, operation: impl FnOnce() -> D) -> (D, Vec<usize>) {
        TEST_WRITE_BITMAP.with(|bitmap| *bitmap.borrow_mut() = Some(vec![0; len]));
        let output = operation();
        let bitmap = TEST_WRITE_BITMAP.with(|bitmap| bitmap.borrow_mut().take().unwrap());
        (output, bitmap)
    }

    #[test]
    fn domain_writer_covers_both_sides_and_one_sided_regions_once() {
        // What: ordered column slabs cover multiple canonical regions exactly
        // once when sectors exist on both inputs or only one input.
        let copies = [
            copy(0, 0, 0, [2, 1], [1, 2], 2, 0..6, false),
            copy(1, 0, 2, [2, 2], [1, 2], 2, 0..6, false),
            copy(0, 2, 6, [2, 1], [1, 2], 2, 6..8, false),
            copy(1, 4, 8, [2, 2], [1, 2], 2, 8..12, false),
        ];
        let (output, bitmap) = with_bitmap(12, || {
            try_cat_owned_raw(
                12,
                OwnedCatSide::Domain,
                &copies,
                [&[1., 2., 7., 8.], &[3., 4., 5., 6., 9., 10., 11., 12.]],
            )
            .unwrap()
        });
        assert_eq!(output, [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]);
        assert_eq!(bitmap, [1; 12]);
    }

    #[test]
    fn codomain_writer_covers_interleaved_row_slabs_once() {
        // What: row slabs partition every column of a canonical region without
        // overlap, including a following right-only sector.
        let copies = [
            copy(0, 0, 0, [1, 2], [1, 1], 3, 0..6, false),
            copy(1, 0, 1, [2, 2], [1, 2], 3, 0..6, false),
            copy(1, 4, 6, [2, 2], [1, 2], 2, 6..10, false),
        ];
        let (output, bitmap) = with_bitmap(10, || {
            try_cat_owned_raw(
                10,
                OwnedCatSide::Codomain,
                &copies,
                [&[1., 4.], &[2., 3., 5., 6., 7., 8., 9., 10.]],
            )
            .unwrap()
        });
        assert_eq!(output, [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]);
        assert_eq!(bitmap, [1; 10]);
    }

    #[test]
    fn mapped_writer_transposes_and_conjugates_each_value_once() {
        // What: non-contiguous adjoint source traversal applies conjugation
        // exactly once while still overwriting every destination address once.
        let copies = [copy(0, 0, 0, [2, 2], [2, 1], 2, 0..4, true)];
        let source = [
            Complex64::new(1., 1.),
            Complex64::new(2., 2.),
            Complex64::new(3., 3.),
            Complex64::new(4., 4.),
        ];
        let (output, bitmap) = with_bitmap(4, || {
            try_cat_owned_raw(4, OwnedCatSide::Domain, &copies, [&source, &[]]).unwrap()
        });
        assert_eq!(
            output,
            [
                Complex64::new(1., -1.),
                Complex64::new(3., -3.),
                Complex64::new(2., -2.),
                Complex64::new(4., -4.),
            ]
        );
        assert_eq!(bitmap, [1; 4]);
    }

    #[test]
    fn mixed_writer_widens_and_conjugates_without_staging() {
        // What: mixed real/complex slabs write directly into the final complex
        // payload and complex conjugation is applied exactly once.
        let copies = [
            copy(0, 0, 0, [2, 1], [1, 2], 2, 0..4, false),
            copy(1, 0, 2, [2, 1], [1, 2], 2, 0..4, true),
        ];
        let complex = [Complex64::new(3., 4.), Complex64::new(5., 6.)];
        let (output, bitmap) = with_bitmap(4, || {
            try_cat_owned_c64_raw(
                4,
                OwnedCatSide::Domain,
                &copies,
                [
                    OwnedCatC64Source::F64(&[1., 2.]),
                    OwnedCatC64Source::C64(&complex),
                ],
            )
            .unwrap()
        });
        assert_eq!(
            output,
            [
                Complex64::new(1., 0.),
                Complex64::new(2., 0.),
                Complex64::new(3., -4.),
                Complex64::new(5., -6.),
            ]
        );
        assert_eq!(bitmap, [1; 4]);
    }

    #[test]
    fn empty_and_zero_extent_outputs_are_completed() {
        // What: empty canonical storage and explicit zero-extent copy geometry
        // complete without manufacturing a nonempty payload.
        assert_eq!(
            try_cat_owned_raw::<f64>(0, OwnedCatSide::Domain, &[], [&[], &[]]),
            Some(vec![])
        );
        let zero = [copy(0, 0, 0, [0, 1], [1, 0], 0, 0..0, false)];
        assert_eq!(
            try_cat_owned_raw::<f64>(0, OwnedCatSide::Domain, &zero, [&[], &[]]),
            Some(vec![])
        );
    }

    #[test]
    fn invalid_geometry_declines_before_opening_the_transaction() {
        // What: overlap with equal total area, holes, padding, reordered or
        // nonmonotone slabs, source OOB, and checked overflow are all declined.
        let valid_sources = [&[1., 2.][..], &[3., 4.][..]];
        let invalid = [
            (
                4,
                vec![
                    copy(0, 0, 0, [2, 1], [1, 2], 2, 0..4, false),
                    copy(1, 0, 0, [2, 1], [1, 2], 2, 0..4, false),
                ],
            ),
            (
                6,
                vec![
                    copy(0, 0, 0, [2, 1], [1, 2], 2, 0..6, false),
                    copy(1, 0, 4, [2, 1], [1, 2], 2, 0..6, false),
                ],
            ),
            (4, vec![copy(0, 0, 0, [2, 1], [1, 2], 2, 0..2, false)]),
            (
                4,
                vec![
                    copy(1, 0, 0, [2, 1], [1, 2], 2, 0..4, false),
                    copy(0, 0, 2, [2, 1], [1, 2], 2, 0..4, false),
                ],
            ),
            (
                4,
                vec![
                    copy(0, 0, 2, [2, 1], [1, 2], 2, 2..4, false),
                    copy(1, 0, 0, [2, 1], [1, 2], 2, 0..2, false),
                ],
            ),
            (2, vec![copy(0, 1, 0, [2, 1], [1, 2], 2, 0..2, false)]),
            (
                2,
                vec![copy(
                    0,
                    usize::MAX,
                    0,
                    [2, 1],
                    [usize::MAX, 1],
                    2,
                    0..2,
                    false,
                )],
            ),
        ];
        for (required_len, copies) in invalid {
            assert!(
                try_cat_owned_raw(required_len, OwnedCatSide::Domain, &copies, valid_sources)
                    .is_none()
            );
            assert!(TEST_WRITE_BITMAP.with(|bitmap| bitmap.borrow().is_none()));
        }
    }

    #[test]
    fn codomain_overlapping_row_slabs_decline_before_writing() {
        // What: overlapping codomain row slabs are rejected before the owned
        // transaction writes any physical destination address.
        let copies = [
            copy(0, 0, 0, [2, 2], [1, 2], 3, 0..6, false),
            copy(1, 0, 1, [1, 2], [1, 1], 3, 0..6, false),
        ];
        TEST_WRITE_BITMAP.with(|bitmap| *bitmap.borrow_mut() = Some(vec![0; 6]));

        let output = try_cat_owned_raw(
            6,
            OwnedCatSide::Codomain,
            &copies,
            [&[1., 2., 3., 4.], &[5., 6.]],
        );

        assert!(output.is_none());
        let bitmap = TEST_WRITE_BITMAP.with(|bitmap| bitmap.borrow_mut().take().unwrap());
        assert_eq!(bitmap, [0; 6]);
    }

    #[derive(Clone, Copy)]
    struct PanicValue;

    impl strided_kernel::ElementOpApply for PanicValue {}

    impl ConjugateValue for PanicValue {
        fn maybe_conj(self, _conjugate: bool) -> Self {
            panic!("injected mapped-write panic")
        }
    }

    #[test]
    fn panic_after_a_partial_mapped_write_cannot_return_a_vector() {
        // What: unwinding inside an accepted mapped write leaves the owned
        // transaction at length zero, so no partial output can escape.
        let copies = [copy(0, 0, 0, [2, 1], [1, 2], 2, 0..2, true)];
        let result = std::panic::catch_unwind(|| {
            let _ = try_cat_owned_raw(
                2,
                OwnedCatSide::Domain,
                &copies,
                [&[PanicValue, PanicValue], &[]],
            );
        });
        assert!(result.is_err());
    }
}
