//! Independent expectation for the truncated-factorization compositions
//! (`svd_compact` / `eigh_full` / `eig_full` → `diagview` → `find_truncated`
//! → `restrict_leg` / `restrict_diagonal`, #1300, #1534).
//!
//! Nothing here calls a TeNeT factorization or a TeNeT truncation decision:
//!
//! * [`sector_matrices!`] reads each coupled sector's reduced matrix (TensorKit
//!   `block(t, c)`) from the tensor's public blocks. Rows are grouped by
//!   codomain tree and columns by domain tree, both in first-appearance
//!   order, and a block's multi-index is linearized first-axis-fastest. The
//!   arrangement is arbitrary; spectra do not depend on it, and a
//!   reconstruction is compared only with another matrix read the same way.
//! * [`svd`] is a one-sided (Hestenes) Jacobi SVD written here, and
//!   [`hermitian_eigen`] reads signed eigenvalues from it by Rayleigh
//!   quotients. [`triangular_eigenvalues`] is the hand answer for the
//!   triangular fixtures the general eigendecomposition is tested on.
//! * [`select`] is a hand implementation of each policy's documented rule
//!   (`tenet_matrixalgebra::Truncation`), written over a flat sorted candidate
//!   list rather than `select_truncation`'s heap.

#![allow(dead_code)]

use num_complex::Complex64;
use tenet::core::{ProductSector, SU2Irrep, U1Irrep, Z2Irrep};

/// Closed-form quantum dimension of each multiplicity-free sector type these
/// tests use, so no weight is read back from TeNeT: 1 for the abelian labels,
/// `2j + 1` for SU(2), the product of the parts for a product sector.
pub trait ClosedFormDim {
    fn closed_form_dim(&self) -> f64;
}

impl ClosedFormDim for U1Irrep {
    fn closed_form_dim(&self) -> f64 {
        1.0
    }
}

impl ClosedFormDim for Z2Irrep {
    fn closed_form_dim(&self) -> f64 {
        1.0
    }
}

impl ClosedFormDim for SU2Irrep {
    fn closed_form_dim(&self) -> f64 {
        (self.twice_spin() + 1) as f64
    }
}

impl<L: ClosedFormDim, R: ClosedFormDim> ClosedFormDim for ProductSector<L, R> {
    fn closed_form_dim(&self) -> f64 {
        self.left().closed_form_dim() * self.right().closed_form_dim()
    }
}

/// A dense column-major complex matrix.
#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Complex64>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![Complex64::new(0.0, 0.0); rows * cols],
        }
    }

    pub fn at(&self, row: usize, col: usize) -> Complex64 {
        self.data[row + col * self.rows]
    }

    fn col(&self, col: usize) -> &[Complex64] {
        &self.data[col * self.rows..(col + 1) * self.rows]
    }

    pub fn max_abs(&self) -> f64 {
        self.data.iter().map(|v| v.norm()).fold(0.0, f64::max)
    }
}

/// One stored block, with its values listed first-axis-fastest over `shape`.
pub struct BlockEntry<S, K> {
    pub coupled: S,
    pub row: K,
    pub col: K,
    pub shape: Vec<usize>,
    pub values: Vec<Complex64>,
}

/// Every multi-index of `shape`, first axis fastest.
pub fn multi_indices(shape: &[usize]) -> Vec<Vec<usize>> {
    let total: usize = shape.iter().product();
    (0..total)
        .map(|mut flat| {
            shape
                .iter()
                .map(|&extent| {
                    let index = flat % extent;
                    flat /= extent;
                    index
                })
                .collect()
        })
        .collect()
}

/// Stacks the blocks of each coupled sector into its reduced matrix.
pub fn assemble<S: PartialEq + Clone, K: PartialEq + Clone>(
    codomain_rank: usize,
    blocks: Vec<BlockEntry<S, K>>,
) -> Vec<(S, Matrix)> {
    // Per sector: the row and column groups with their offsets and sizes.
    struct Groups<S, K> {
        sector: S,
        rows: Vec<(K, usize, usize)>,
        cols: Vec<(K, usize, usize)>,
    }
    fn place<K: PartialEq + Clone>(groups: &mut Vec<(K, usize, usize)>, key: &K, size: usize) {
        if !groups.iter().any(|(k, _, _)| k == key) {
            let offset = groups.last().map_or(0, |(_, o, s)| o + s);
            groups.push((key.clone(), offset, size));
        }
    }
    let mut sectors: Vec<Groups<S, K>> = Vec::new();
    for block in &blocks {
        let rows: usize = block.shape[..codomain_rank].iter().product();
        let cols: usize = block.shape[codomain_rank..].iter().product();
        let index = match sectors.iter().position(|g| g.sector == block.coupled) {
            Some(index) => index,
            None => {
                sectors.push(Groups {
                    sector: block.coupled.clone(),
                    rows: Vec::new(),
                    cols: Vec::new(),
                });
                sectors.len() - 1
            }
        };
        place(&mut sectors[index].rows, &block.row, rows);
        place(&mut sectors[index].cols, &block.col, cols);
    }
    let mut matrices: Vec<(S, Matrix)> = sectors
        .iter()
        .map(|g| {
            let rows = g.rows.last().map_or(0, |(_, o, s)| o + s);
            let cols = g.cols.last().map_or(0, |(_, o, s)| o + s);
            (g.sector.clone(), Matrix::zeros(rows, cols))
        })
        .collect();
    for block in blocks {
        let index = sectors
            .iter()
            .position(|g| g.sector == block.coupled)
            .unwrap();
        let g = &sectors[index];
        let &(_, row0, block_rows) = g.rows.iter().find(|(k, _, _)| *k == block.row).unwrap();
        let &(_, col0, _) = g.cols.iter().find(|(k, _, _)| *k == block.col).unwrap();
        let matrix = &mut matrices[index].1;
        for (flat, value) in block.values.into_iter().enumerate() {
            let (row, col) = (flat % block_rows.max(1), flat / block_rows.max(1));
            let rows = matrix.rows;
            matrix.data[(row0 + row) + (col0 + col) * rows] = value;
        }
    }
    matrices
}

/// Each coupled sector's reduced matrix of a tensor, as `Vec<(sector,
/// Matrix)>`. The including crate declares `numerics` at its root.
#[macro_export]
macro_rules! sector_matrices {
    ($tensor:expr) => {{
        let tensor = &$tensor;
        let blocks = tensor
            .blocks()
            .unwrap()
            .map(|(trees, view)| {
                let shape = view.shape().to_vec();
                let values = $crate::truncation_oracle::multi_indices(&shape)
                    .iter()
                    .map(|index| $crate::numerics::Numeric::wide(*view.get(index).unwrap()))
                    .collect();
                $crate::truncation_oracle::BlockEntry {
                    coupled: trees.coupled().clone(),
                    row: (
                        trees.codomain_uncoupled().to_vec(),
                        trees.codomain_innerlines().to_vec(),
                        trees.codomain_vertices().to_vec(),
                    ),
                    col: (
                        trees.domain_uncoupled().to_vec(),
                        trees.domain_innerlines().to_vec(),
                        trees.domain_vertices().to_vec(),
                    ),
                    shape,
                    values,
                }
            })
            .collect();
        $crate::truncation_oracle::assemble(tensor.codomain_rank(), blocks)
    }};
}

/// The matrix of `sector` in a [`sector_matrices!`] result.
pub fn matrix_of<'a, S: PartialEq>(matrices: &'a [(S, Matrix)], sector: &S) -> &'a Matrix {
    &matrices
        .iter()
        .find(|(s, _)| s == sector)
        .expect("sector present")
        .1
}

/// A singular value decomposition `a = sum_k values[k] * left_k * right_k^H`,
/// sorted by descending singular value.
#[derive(Clone, Debug)]
pub struct Svd {
    pub values: Vec<f64>,
    /// `rows x n` left singular vectors (a zero column for a zero value).
    pub left: Matrix,
    /// `n x n` right singular vectors.
    pub right: Matrix,
}

/// One-sided Jacobi SVD: rotate column pairs of `a` (and of the identity) until
/// every pair is orthogonal; the column norms are then the singular values.
pub fn svd(a: &Matrix) -> Svd {
    let (m, n) = (a.rows, a.cols);
    let mut u = a.clone();
    let mut v = Matrix::zeros(n, n);
    for k in 0..n {
        v.data[k + k * n] = Complex64::new(1.0, 0.0);
    }
    let rotate = |x: &mut Matrix, rows: usize, i: usize, j: usize, c: f64, s: f64, w: Complex64| {
        for k in 0..rows {
            let xi = x.data[k + i * rows];
            let xj = x.data[k + j * rows] * w.conj();
            x.data[k + i * rows] = xi * c - xj * s;
            x.data[k + j * rows] = xi * s + xj * c;
        }
    };
    for _sweep in 0..100 {
        let mut rotated = false;
        for i in 0..n {
            for j in i + 1..n {
                let (ci, cj) = (u.col(i), u.col(j));
                let alpha: f64 = ci.iter().map(|z| z.norm_sqr()).sum();
                let beta: f64 = cj.iter().map(|z| z.norm_sqr()).sum();
                let gamma: Complex64 = ci.iter().zip(cj).map(|(x, y)| x.conj() * y).sum();
                let g = gamma.norm();
                if g <= 4.0 * f64::EPSILON * (alpha * beta).sqrt() || g == 0.0 {
                    continue;
                }
                rotated = true;
                let w = gamma / g;
                let zeta = (beta - alpha) / (2.0 * g);
                let t = if zeta == 0.0 {
                    1.0
                } else {
                    zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                rotate(&mut u, m, i, j, c, s, w);
                rotate(&mut v, n, i, j, c, s, w);
            }
        }
        if !rotated {
            break;
        }
    }
    let mut order: Vec<(f64, usize)> = (0..n)
        .map(|k| (u.col(k).iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt(), k))
        .collect();
    order.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut left = Matrix::zeros(m, n);
    let mut right = Matrix::zeros(n, n);
    for (slot, &(sigma, k)) in order.iter().enumerate() {
        for row in 0..m {
            if sigma > 0.0 {
                left.data[row + slot * m] = u.data[row + k * m] / sigma;
            }
        }
        for row in 0..n {
            right.data[row + slot * n] = v.data[row + k * n];
        }
    }
    Svd {
        values: order.iter().map(|&(sigma, _)| sigma).collect(),
        left,
        right,
    }
}

/// `sum_{k < kept} values[k] * left_k * right_k^H`.
pub fn svd_truncated(svd: &Svd, kept: usize) -> Matrix {
    let (m, n) = (svd.left.rows, svd.right.rows);
    let mut out = Matrix::zeros(m, n);
    for k in 0..kept {
        for col in 0..n {
            let right = svd.right.at(col, k).conj() * svd.values[k];
            for row in 0..m {
                out.data[row + col * m] += svd.left.at(row, k) * right;
            }
        }
    }
    out
}

/// Signed eigenvalues and eigenvectors of a Hermitian matrix, by descending
/// magnitude: for distinct `|lambda|`, the right singular vectors are the
/// eigenvectors and `lambda = v^H a v`.
pub fn hermitian_eigen(a: &Matrix) -> (Vec<f64>, Matrix) {
    let decomposition = svd(a);
    let n = a.cols;
    let values = (0..n)
        .map(|k| {
            let v = decomposition.right.col(k);
            let mut quotient = Complex64::new(0.0, 0.0);
            for col in 0..n {
                for row in 0..n {
                    quotient += v[row].conj() * a.at(row, col) * v[col];
                }
            }
            quotient.re
        })
        .collect();
    (values, decomposition.right)
}

/// `sum_{k < kept} values[k] * v_k * v_k^H`.
pub fn eigen_truncated(values: &[f64], vectors: &Matrix, kept: usize) -> Matrix {
    let n = vectors.rows;
    let mut out = Matrix::zeros(n, n);
    for (k, &value) in values.iter().enumerate().take(kept) {
        for col in 0..n {
            let right = vectors.at(col, k).conj() * value;
            for row in 0..n {
                out.data[row + col * n] += vectors.at(row, k) * right;
            }
        }
    }
    out
}

/// The eigenvalues of a triangular matrix are its diagonal. Panics unless `a`
/// is exactly upper or lower triangular, so a fixture cannot drift out of the
/// case this answer covers. Sorted by descending magnitude.
pub fn triangular_eigenvalues(a: &Matrix) -> Vec<Complex64> {
    assert_eq!(a.rows, a.cols, "an endomorphism block");
    let zero = |strict_lower: bool| {
        (0..a.rows).all(|row| {
            (0..a.cols)
                .filter(|&col| if strict_lower { row > col } else { row < col })
                .all(|col| a.at(row, col) == Complex64::new(0.0, 0.0))
        })
    };
    assert!(
        zero(true) || zero(false),
        "the eig fixtures must be triangular"
    );
    let mut values: Vec<Complex64> = (0..a.rows).map(|k| a.at(k, k)).collect();
    values.sort_by(|x, y| y.norm().total_cmp(&x.norm()));
    values
}

/// One coupled sector's offered spectrum: quantum dimension and descending
/// magnitudes.
#[derive(Clone, Debug)]
pub struct Offer<S> {
    pub sector: S,
    pub dim: f64,
    pub magnitudes: Vec<f64>,
}

/// A truncation policy, restated for the hand selection.
#[derive(Clone, Debug)]
pub enum Policy<S> {
    Full,
    Rank(usize),
    /// Keep `v >= rtol * sqrt(sum dim v^2)`.
    RelativeCutoff(f64),
    /// Keep `v >= rtol * max v`.
    RelativeInfCutoff(f64),
    /// Discard ascending while `sum dim v^2 <= (rtol * norm)^2`.
    RelativeError(f64),
    /// A fixed per-sector prefix (absent sector: zero).
    Space(Vec<(S, usize)>),
    /// Both components must keep a value.
    And(Box<Policy<S>>, Box<Policy<S>>),
}

fn weighted_norm<S>(offers: &[Offer<S>]) -> f64 {
    offers
        .iter()
        .map(|o| o.dim * o.magnitudes.iter().map(|v| v * v).sum::<f64>())
        .sum::<f64>()
        .sqrt()
}

/// Kept prefix length per offer. An exact cross-sector tie goes to the sector
/// that sorts first under `S: Ord`, which is TensorKit's order for every
/// fixture that has a tie (non-negative U(1) charges, SU(2) spins).
pub fn select<S: Ord + Clone>(offers: &[Offer<S>], policy: &Policy<S>) -> Vec<usize> {
    match policy {
        Policy::Full => offers.iter().map(|o| o.magnitudes.len()).collect(),
        Policy::Rank(rank) => {
            let mut candidates: Vec<(f64, &S, usize, usize)> = offers
                .iter()
                .enumerate()
                .flat_map(|(i, o)| {
                    o.magnitudes
                        .iter()
                        .enumerate()
                        .map(move |(p, &v)| (v, &o.sector, p, i))
                })
                .collect();
            candidates.sort_by(|a, b| {
                b.0.total_cmp(&a.0)
                    .then_with(|| a.1.cmp(b.1))
                    .then(a.2.cmp(&b.2))
            });
            let mut kept = vec![0; offers.len()];
            let mut used = 0.0;
            for (_, _, _, i) in candidates {
                if used + offers[i].dim > *rank as f64 + 1e-12 {
                    break;
                }
                used += offers[i].dim;
                kept[i] += 1;
            }
            kept
        }
        Policy::RelativeCutoff(rtol) => {
            let threshold = rtol * weighted_norm(offers);
            offers
                .iter()
                .map(|o| o.magnitudes.iter().filter(|&&v| v >= threshold).count())
                .collect()
        }
        Policy::RelativeInfCutoff(rtol) => {
            let max = offers
                .iter()
                .flat_map(|o| o.magnitudes.iter().copied())
                .fold(0.0, f64::max);
            offers
                .iter()
                .map(|o| o.magnitudes.iter().filter(|&&v| v >= rtol * max).count())
                .collect()
        }
        Policy::RelativeError(rtol) => {
            let bound = rtol * weighted_norm(offers);
            let values: usize = offers.iter().map(|o| o.magnitudes.len()).sum();
            let limit = bound * bound * (1.0 + (values + 5) as f64 * f64::EPSILON);
            let mut candidates: Vec<(f64, &S, usize, usize)> = offers
                .iter()
                .enumerate()
                .flat_map(|(i, o)| {
                    o.magnitudes
                        .iter()
                        .enumerate()
                        .map(move |(p, &v)| (v, &o.sector, p, i))
                })
                .collect();
            candidates.sort_by(|a, b| {
                a.0.total_cmp(&b.0)
                    .then_with(|| a.1.cmp(b.1))
                    .then(b.2.cmp(&a.2))
            });
            let mut kept: Vec<usize> = offers.iter().map(|o| o.magnitudes.len()).collect();
            let mut discarded = 0.0;
            for (v, _, _, i) in candidates {
                let next = discarded + offers[i].dim * v * v;
                if next > limit {
                    break;
                }
                discarded = next;
                kept[i] -= 1;
            }
            kept
        }
        Policy::Space(ranks) => offers
            .iter()
            .map(|o| {
                ranks
                    .iter()
                    .find(|(s, _)| *s == o.sector)
                    .map_or(0, |&(_, r)| r)
                    .min(o.magnitudes.len())
            })
            .collect(),
        Policy::And(left, right) => select(offers, left)
            .into_iter()
            .zip(select(offers, right))
            .map(|(a, b)| a.min(b))
            .collect(),
    }
}

/// `sqrt(sum_c dim(c) sum_{discarded} v^2)`.
pub fn discarded_norm<S>(offers: &[Offer<S>], kept: &[usize]) -> f64 {
    offers
        .iter()
        .zip(kept)
        .map(|(o, &k)| o.dim * o.magnitudes[k..].iter().map(|v| v * v).sum::<f64>())
        .sum::<f64>()
        .sqrt()
}

/// The expected kept bond as `(sector, count)` pairs sorted by sector, zero
/// counts dropped.
pub fn kept_pairs<S: Ord + Clone>(offers: &[Offer<S>], kept: &[usize]) -> Vec<(S, usize)> {
    let mut pairs: Vec<(S, usize)> = offers
        .iter()
        .zip(kept)
        .filter(|(_, &k)| k > 0)
        .map(|(o, &k)| (o.sector.clone(), k))
        .collect();
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    pairs
}

/// The policy sweep every composition case runs: `(name, Truncation, Policy)`.
/// `$target` is the fixture's `truncspace` target.
#[macro_export]
macro_rules! policies {
    ($target:expr) => {{
        use tenet::prelude::Truncation;
        use $crate::truncation_oracle::Policy;
        let target = &$target;
        let pairs: Vec<_> = target
            .sectors()
            .unwrap()
            .into_iter()
            .zip(target.degeneracies().iter().copied())
            .collect();
        vec![
            ("Full", Truncation::Full, Policy::Full),
            ("Rank(huge)", Truncation::rank(4096), Policy::Rank(4096)),
            ("Rank(4)", Truncation::rank(4), Policy::Rank(4)),
            ("Rank(3)", Truncation::rank(3), Policy::Rank(3)),
            ("Rank(1)", Truncation::rank(1), Policy::Rank(1)),
            ("Rank(0)", Truncation::rank(0), Policy::Rank(0)),
            (
                "Tolerance",
                Truncation::relative_cutoff(0.25).unwrap(),
                Policy::RelativeCutoff(0.25),
            ),
            (
                "ToleranceInf",
                Truncation::relative_inf_cutoff(0.4).unwrap(),
                Policy::RelativeInfCutoff(0.4),
            ),
            (
                "DiscardWeight",
                Truncation::relative_error(0.2).unwrap(),
                Policy::RelativeError(0.2),
            ),
            (
                "Space",
                Truncation::space(target.truncspace()),
                Policy::Space(pairs.clone()),
            ),
            (
                "Space & Rank",
                Truncation::space(target.truncspace()).and(Truncation::rank(2)),
                Policy::And(Box::new(Policy::Space(pairs)), Box::new(Policy::Rank(2))),
            ),
        ]
    }};
}

/// The kept bond leg is exactly the hand selection: same `(sector, count)`
/// pairs, not dual.
#[macro_export]
macro_rules! assert_kept_bond {
    ($bond:expr, $offers:expr, $kept:expr, $case:expr) => {{
        let bond = &$bond;
        let want = $crate::truncation_oracle::kept_pairs(&$offers, &$kept);
        let mut got: Vec<_> = bond
            .sectors()
            .unwrap()
            .into_iter()
            .zip(bond.degeneracies().iter().copied())
            .collect();
        got.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(got, want, "{}: kept bond", $case);
        assert!(!bond.is_dual(), "{}: the bond is not dual", $case);
    }};
}

/// A factor's block geometry is the canonical layout of its own homspace,
/// i.e. that of a fresh `zeros` on the same spaces.
#[macro_export]
macro_rules! assert_canonical_layout {
    ($tensor:expr, $what:expr) => {{
        let got = &$tensor;
        let want: tenet::prelude::TensorMap<_, f64> = tenet::prelude::TensorMap::zeros(
            got.runtime(),
            got.codomain().iter(),
            got.domain().iter(),
        )
        .unwrap();
        assert_eq!(
            got.block_count(),
            want.block_count(),
            "{} block count",
            $what
        );
        for index in 0..want.block_count() {
            let (left, right) = (got.block(index).unwrap(), want.block(index).unwrap());
            assert_eq!(left.key(), right.key(), "{} block {index} key", $what);
            assert_eq!(left.shape(), right.shape(), "{} block {index} shape", $what);
            assert_eq!(
                left.strides(),
                right.strides(),
                "{} block {index} strides",
                $what
            );
            assert_eq!(
                left.offset(),
                right.offset(),
                "{} block {index} offset",
                $what
            );
        }
    }};
}
