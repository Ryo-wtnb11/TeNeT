//! `TensorMap::block(&c)` / `blocks()` (TensorKit `block(t, c)` /
//! `blocks(t)`) against oracles that never read the coupled payload layout:
//!
//! - the label oracle: a tensor filled by `from_block_fn` with a function of
//!   its tree labels and local indices, assembled into sector matrices from
//!   the subblock labels and the legs' degeneracies;
//! - the physical dense expansion (U(1), SU(2)): `M ≅ ⊕_c B_c ⊗ 1_{dim c}`
//!   unitarily, so `tr((M†M)^k) = Σ_c dim(c) tr((B_c†B_c)^k)`;
//! - composition: `block(A∘B, c) = block(A, c) · block(B, c)`;
//! - the lazy adjoint `block(t', c) = block(t, c)'` over the parent's payload.
//!
//! The CUDA gate runs with `cargo test -p tenet-rs --no-default-features
//! --features cuda,cpu-faer --test coupled_blocks -- --ignored` on a CUDA host.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};

use num_complex::Complex64;
use tenet::core::{HostReadableStorage, TypedSectorAdmission};
use tenet::prelude::*;
use tenet::typed::{BlockFusionTrees, CoupledBlock, CoupledBlockPayload};

trait Val: TensorScalar + Copy + PartialEq + Debug {
    fn make(hash: u64) -> Self;
    fn c(self) -> Complex64;
}

impl Val for f64 {
    fn make(hash: u64) -> Self {
        ((hash % 2001) as f64 - 1000.0) / 257.0
    }
    fn c(self) -> Complex64 {
        Complex64::new(self, 0.0)
    }
}

impl Val for Complex64 {
    fn make(hash: u64) -> Self {
        Complex64::new(
            f64::make(hash),
            f64::make(hash.rotate_left(29) ^ 0x9E37_79B9),
        )
    }
    fn c(self) -> Complex64 {
        self
    }
}

/// A value fixed by the tree labels and local index alone.
fn label_value<S: Debug, D: Val>(trees: &BlockFusionTrees<S>, index: &[usize]) -> D {
    let mut hasher = DefaultHasher::new();
    format!("{trees:?}").hash(&mut hasher);
    index.hash(&mut hasher);
    D::make(hasher.finish())
}

/// Column-major dense matrix.
#[derive(Clone, Debug)]
struct Mat {
    rows: usize,
    cols: usize,
    data: Vec<Complex64>,
}

impl Mat {
    fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![Complex64::new(0.0, 0.0); rows * cols],
        }
    }
    fn at(&self, r: usize, c: usize) -> Complex64 {
        self.data[r + c * self.rows]
    }
    fn mul(&self, other: &Mat) -> Mat {
        assert_eq!(self.cols, other.rows);
        let mut out = Mat::zeros(self.rows, other.cols);
        for c in 0..other.cols {
            for k in 0..self.cols {
                let b = other.at(k, c);
                for r in 0..self.rows {
                    out.data[r + c * out.rows] += self.at(r, k) * b;
                }
            }
        }
        out
    }
    fn adjoint(&self) -> Mat {
        let mut out = Mat::zeros(self.cols, self.rows);
        for c in 0..self.cols {
            for r in 0..self.rows {
                out.data[c + r * out.rows] = self.at(r, c).conj();
            }
        }
        out
    }
    fn trace(&self) -> Complex64 {
        (0..self.rows.min(self.cols)).map(|i| self.at(i, i)).sum()
    }
    fn gram_moments(&self) -> [Complex64; 3] {
        let g = self.adjoint().mul(self);
        let g2 = g.mul(&g);
        [g.trace(), g2.trace(), g2.mul(&g).trace()]
    }
}

fn read<D: Val, S: HostReadableStorage<D>>(block: &CoupledBlock<'_, D, S>) -> Mat {
    let mut out = Mat::zeros(block.rows(), block.cols());
    for c in 0..block.cols() {
        for r in 0..block.rows() {
            out.data[r + c * block.rows()] = block.get(r, c).unwrap().c();
        }
    }
    assert!(block.get(block.rows(), 0).is_none());
    assert!(block.get(0, block.cols()).is_none());
    out
}

fn assert_mat_close(what: &str, got: &Mat, want: &Mat) {
    assert_eq!(
        (got.rows, got.cols),
        (want.rows, want.cols),
        "{what}: shape"
    );
    let scale = want.data.iter().map(|z| z.norm()).fold(1.0, f64::max);
    for (g, w) in got.data.iter().zip(&want.data) {
        assert!(
            (g - w).norm() <= 1e-12 * scale * want.rows.max(1) as f64,
            "{what}: {g} vs {w}"
        );
    }
}

fn product(values: impl IntoIterator<Item = usize>) -> usize {
    values.into_iter().product()
}

/// Expected sector matrices of `t`, assembled only from its subblock labels,
/// its legs' degeneracies and `value`. Row trees are the codomain trees in
/// first-appearance order over `subblocks()`, column trees likewise; each tree
/// spans its degeneracy extent, column-major in its legs.
fn expected_blocks<R, D>(
    t: &TensorMap<R, D>,
    mut value: impl FnMut(&BlockFusionTrees<R::Sector>, &[usize]) -> Complex64,
) -> Vec<(R::Sector, Mat)>
where
    R: TypedSectorAdmission,
    R::Mode: tenet::typed::TypedTensorModeDispatch<R>,
    D: Val,
{
    let codomain = t.codomain();
    let domain = t.domain();
    let dims = |legs: &[GradedSpace<R>], sectors: &[R::Sector]| -> Vec<usize> {
        legs.iter()
            .zip(sectors)
            .map(|(leg, sector)| leg.degeneracy(sector).unwrap())
            .collect()
    };
    struct Sector<S> {
        label: S,
        rows: Vec<(String, Vec<usize>)>,
        cols: Vec<(String, Vec<usize>)>,
        pairs: Vec<(usize, usize, BlockFusionTrees<S>)>,
    }
    let mut sectors: Vec<Sector<R::Sector>> = Vec::new();
    for index in 0..t.subblock_count() {
        let trees = t.subblock_fusion_trees(index).unwrap();
        let row_key = format!(
            "{:?}{:?}{:?}",
            trees.codomain_uncoupled(),
            trees.codomain_innerlines(),
            trees.codomain_vertices()
        );
        let col_key = format!(
            "{:?}{:?}{:?}",
            trees.domain_uncoupled(),
            trees.domain_innerlines(),
            trees.domain_vertices()
        );
        let position = match sectors.iter().position(|s| &s.label == trees.coupled()) {
            Some(position) => position,
            None => {
                sectors.push(Sector {
                    label: trees.coupled().clone(),
                    rows: Vec::new(),
                    cols: Vec::new(),
                    pairs: Vec::new(),
                });
                sectors.len() - 1
            }
        };
        let sector = &mut sectors[position];
        let row = match sector.rows.iter().position(|(key, _)| *key == row_key) {
            Some(row) => row,
            None => {
                sector
                    .rows
                    .push((row_key, dims(&codomain, trees.codomain_uncoupled())));
                sector.rows.len() - 1
            }
        };
        let col = match sector.cols.iter().position(|(key, _)| *key == col_key) {
            Some(col) => col,
            None => {
                sector
                    .cols
                    .push((col_key, dims(&domain, trees.domain_uncoupled())));
                sector.cols.len() - 1
            }
        };
        sector.pairs.push((row, col, trees));
    }

    let offsets = |extents: &[(String, Vec<usize>)]| -> (Vec<usize>, usize) {
        let mut offsets = Vec::new();
        let mut total = 0;
        for (_, dims) in extents {
            offsets.push(total);
            total += product(dims.iter().copied());
        }
        (offsets, total)
    };
    let unravel = |mut linear: usize, dims: &[usize]| -> Vec<usize> {
        dims.iter()
            .map(|&n| {
                let i = linear % n;
                linear /= n;
                i
            })
            .collect()
    };
    sectors
        .into_iter()
        .map(|sector| {
            let (row_offsets, rows) = offsets(&sector.rows);
            let (col_offsets, cols) = offsets(&sector.cols);
            let mut matrix = Mat::zeros(rows, cols);
            for (row, col, trees) in &sector.pairs {
                let row_dims = &sector.rows[*row].1;
                let col_dims = &sector.cols[*col].1;
                for a in 0..product(row_dims.iter().copied()) {
                    for b in 0..product(col_dims.iter().copied()) {
                        let mut local = unravel(a, row_dims);
                        local.extend(unravel(b, col_dims));
                        let r = row_offsets[*row] + a;
                        let c = col_offsets[*col] + b;
                        matrix.data[r + c * rows] = value(trees, &local);
                    }
                }
            }
            (sector.label, matrix)
        })
        .collect()
}

/// `blocks()` and every `block(&c)` equal `expected` exactly, sector for
/// sector and in the same (storage) order.
fn assert_blocks_equal<R, D>(what: &str, t: &TensorMap<R, D>, expected: &[(R::Sector, Mat)])
where
    R: TypedSectorAdmission,
    R::Mode: tenet::typed::TypedTensorModeDispatch<R>,
    D: Val,
{
    let blocks: Vec<_> = t.blocks().unwrap().collect();
    assert_eq!(blocks.len(), expected.len(), "{what}: sector count");
    for ((sector, block), (want_sector, want)) in blocks.iter().zip(expected) {
        assert_eq!(sector, want_sector, "{what}: sector order");
        let got = read(block);
        assert_eq!(got.rows, want.rows, "{what}: rows of {sector:?}");
        assert_eq!(got.cols, want.cols, "{what}: cols of {sector:?}");
        assert_eq!(got.data, want.data, "{what}: entries of {sector:?}");
        assert_eq!(
            read(&t.block(sector).unwrap()).data,
            want.data,
            "{what}: block({sector:?})"
        );
    }
}

/// The subblock values of `t` (pre-existing accessor) placed by labels.
fn subblock_value<'a, R, D>(
    t: &'a TensorMap<R, D>,
) -> impl FnMut(&BlockFusionTrees<R::Sector>, &[usize]) -> Complex64 + 'a
where
    R: TypedSectorAdmission,
    R::Mode: tenet::typed::TypedTensorModeDispatch<R>,
    D: Val,
{
    let views: HashMap<_, _> = t.subblocks().unwrap().collect();
    move |trees, index| views[trees].get(index).unwrap().c()
}

fn label_case<R, D>(
    what: &str,
    rt: &Runtime,
    codomain: &[&GradedSpace<R>],
    domain: &[&GradedSpace<R>],
) -> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: tenet::typed::TypedTensorModeDispatch<R>
        + tenet::typed::TypedTensorConstructionDispatch<R, D>,
    D: Val,
{
    let t = TensorMap::<R, D>::from_block_fn(
        rt,
        codomain.iter().copied(),
        domain.iter().copied(),
        label_value::<R::Sector, D>,
    )
    .unwrap();
    let expected = expected_blocks(&t, |trees, index| {
        label_value::<R::Sector, D>(trees, index).c()
    });
    assert!(
        expected.len() > 1,
        "{what}: fixture must have several coupled sectors"
    );
    assert_blocks_equal(what, &t, &expected);
    t
}

/// The lazy adjoint's blocks are `block(t, c)'` over the parent's own payload,
/// read without materializing; afterwards they also match the materialized
/// adjoint's subblocks.
fn adjoint_case<R, D>(what: &str, t: &TensorMap<R, D>)
where
    R: TypedSectorAdmission + tenet::core::MultiplicityFreeRigidSymbols<Scalar = f64>,
    R::Mode:
        tenet::typed::TypedTensorModeDispatch<R> + tenet::typed::TypedTensorAdjointDispatch<R, D>,
    D: Val,
{
    let lazy = t.adjoint().unwrap();
    adjoint_view_case(what, t, &lazy);
}

fn adjoint_view_case<R, D>(what: &str, t: &TensorMap<R, D>, lazy: &TensorMap<R, D>)
where
    R: TypedSectorAdmission,
    R::Mode: tenet::typed::TypedTensorModeDispatch<R>,
    D: Val,
{
    let parent = t.data().as_ptr();
    for (sector, block) in lazy.blocks().unwrap() {
        let CoupledBlockPayload::Dense {
            storage,
            adjoint: true,
            ..
        } = block.payload()
        else {
            panic!("{what}: a lazy adjoint block must be a flagged parent view");
        };
        assert_eq!(
            storage.as_ptr(),
            parent,
            "{what}: the view borrows the parent payload"
        );
        let want = read(&t.block(&sector).unwrap()).adjoint();
        assert_eq!(
            read(&block).data,
            want.data,
            "{what}: block(t', {sector:?})"
        );
    }
    let expected = expected_blocks(lazy, subblock_value(lazy));
    assert_blocks_equal(&format!("{what} (materialized subblocks)"), lazy, &expected);
}

fn compose_case<R, D>(what: &str, a: &TensorMap<R, D>, b: &TensorMap<R, D>)
where
    R: TypedSectorAdmission,
    R::Mode:
        tenet::typed::TypedTensorModeDispatch<R> + tenet::typed::TypedTensorContractDispatch<R, D>,
    D: Val,
{
    let ab = a.compose(b).unwrap();
    let mut count = 0;
    for (sector, block) in ab.blocks().unwrap() {
        // TensorKit's empty `d₁ × 0` block of a factor makes the product zero.
        let want = match (a.block(&sector), b.block(&sector)) {
            (Ok(a), Ok(b)) => {
                count += 1;
                read(&a).mul(&read(&b))
            }
            _ => Mat::zeros(block.rows(), block.cols()),
        };
        assert_mat_close(
            &format!("{what}: block(A∘B, {sector:?})"),
            &read(&block),
            &want,
        );
    }
    assert!(count > 1, "{what}: several coupled sectors");
}

macro_rules! physical_case {
    ($what:expr, $t:expr, $dim:expr) => {{
        let t = $t;
        let physical = t.to_physical_dense().unwrap();
        let rows: usize = physical.shape[..t.codomain_rank()].iter().product();
        let cols: usize = physical.shape[t.codomain_rank()..].iter().product();
        let m = Mat {
            rows,
            cols,
            data: physical.data.iter().map(|&x| Val::c(x)).collect(),
        };
        let mut want = [Complex64::new(0.0, 0.0); 3];
        for (sector, block) in t.blocks().unwrap() {
            let dim = ($dim)(&sector) as f64;
            for (total, moment) in want.iter_mut().zip(read(&block).gram_moments()) {
                *total += moment * dim;
            }
        }
        for (k, (got, want)) in m.gram_moments().into_iter().zip(want).enumerate() {
            assert!(
                (got - want).norm() <= 1e-10 * want.norm().max(1.0),
                "{}: tr((M†M)^{}) {got} vs Σ dim(c) tr((B†B)^{}) {want}",
                $what,
                k + 1,
                k + 1
            );
        }
    }};
}

fn u1_legs() -> (GradedSpace<U1FusionRule>, GradedSpace<U1FusionRule>) {
    let v = GradedSpace::try_new(
        U1FusionRule,
        [(-1, 2), (0, 1), (1, 3)].map(|(q, n)| (U1Irrep::new(q), n)),
    )
    .unwrap();
    let w = v.try_dual().unwrap();
    (v, w)
}

fn su2_legs() -> (GradedSpace<SU2FusionRule>, GradedSpace<SU2FusionRule>) {
    let v = GradedSpace::try_new(
        SU2FusionRule,
        [(0, 2), (1, 2), (2, 1)].map(|(s, n)| (SU2Irrep::from_twice_spin(s), n)),
    )
    .unwrap();
    let w = v.try_dual().unwrap();
    (v, w)
}

type Fz2U1 = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;

fn fz2u1_legs() -> (GradedSpace<Fz2U1>, GradedSpace<Fz2U1>) {
    let sector = |q: i32| {
        let parity = if q.rem_euclid(2) == 0 {
            Z2Irrep::EVEN
        } else {
            Z2Irrep::ODD
        };
        ProductSector::new(parity, U1Irrep::new(q))
    };
    let v = GradedSpace::try_new(
        Fz2U1::new(FermionParityFusionRule, U1FusionRule),
        [(-1, 2), (0, 1), (1, 2), (2, 1)].map(|(q, n)| (sector(q), n)),
    )
    .unwrap();
    let w = v.try_dual().unwrap();
    (v, w)
}

/// Label, adjoint and composition oracles over `[v, w] ← [v]` and
/// `[v] ← [v, w]` (`w` dual), for both payload dtypes.
macro_rules! multiplicity_free {
    ($rt:expr, $legs:expr, $name:expr, |$t:ident| $extra:expr) => {{
        let (v, w) = $legs;
        {
            let a = label_case::<_, f64>(concat!($name, " f64"), $rt, &[&v, &w], &[&v]);
            let b = label_case::<_, f64>(concat!($name, " f64 B"), $rt, &[&v], &[&v, &w]);
            adjoint_case(concat!($name, " f64 adjoint"), &a);
            compose_case(concat!($name, " f64 compose"), &a, &b);
            let $t = &a;
            $extra;
        }
        {
            let a = label_case::<_, Complex64>(concat!($name, " c64"), $rt, &[&v, &w], &[&v]);
            let b = label_case::<_, Complex64>(concat!($name, " c64 B"), $rt, &[&v], &[&v, &w]);
            adjoint_case(concat!($name, " c64 adjoint"), &a);
            compose_case(concat!($name, " c64 compose"), &a, &b);
            let $t = &a;
            $extra;
        }
    }};
}

#[test]
fn u1_blocks_match_label_physical_adjoint_and_composition_oracles() {
    let rt = Runtime::builder().build().unwrap();
    multiplicity_free!(&rt, u1_legs(), "U(1)", |t| physical_case!(
        "U(1)",
        t,
        |_: &U1Irrep| 1
    ));
}

#[test]
fn su2_blocks_match_label_physical_adjoint_and_composition_oracles() {
    let rt = Runtime::builder().build().unwrap();
    multiplicity_free!(&rt, su2_legs(), "SU(2)", |t| physical_case!(
        "SU(2)",
        t,
        |c: &SU2Irrep| c.twice_spin() + 1
    ));
}

#[test]
fn fz2u1_blocks_match_label_adjoint_and_composition_oracles() {
    let rt = Runtime::builder().build().unwrap();
    multiplicity_free!(&rt, fz2u1_legs(), "fZ2xU(1)", |t| {
        let _ = t;
    });
}

#[test]
fn compact_diagonal_block_is_its_stored_spectrum() {
    let rt = Runtime::builder().build().unwrap();
    let (v, _) = u1_legs();
    let spectra = [
        (-1, vec![1.5, -2.0]),
        (0, vec![0.25]),
        (1, vec![3.0, -0.5, 4.0]),
    ];
    let d = TensorMap::<_, Complex64>::diagonal(
        &rt,
        &v,
        spectra.iter().map(|(q, values)| SectorSpectrum {
            sector: U1Irrep::new(*q),
            values: values
                .iter()
                .map(|&x| Complex64::new(x, -x / 2.0))
                .collect(),
        }),
    )
    .unwrap();
    for (q, values) in &spectra {
        let block = d.block(&U1Irrep::new(*q)).unwrap();
        let CoupledBlockPayload::Diagonal(stored) = block.payload() else {
            panic!("a compact diagonal block must borrow its spectrum");
        };
        assert_eq!(stored.len(), values.len());
        assert_eq!((block.rows(), block.cols()), (values.len(), values.len()));
        for i in 0..values.len() {
            for j in 0..values.len() {
                let want = if i == j {
                    Complex64::new(values[i], -values[i] / 2.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                assert_eq!(block.get(i, j), Some(want));
            }
        }
    }
    assert_eq!(d.blocks().unwrap().len(), spectra.len());
}

#[test]
fn absent_or_foreign_coupled_sector_is_an_error() {
    let rt = Runtime::builder().build().unwrap();
    let (v, w) = u1_legs();
    let t = TensorMap::<_, f64>::from_block_fn(&rt, [&v, &w], [&v], label_value).unwrap();
    assert!(t.block(&U1Irrep::new(7)).is_err());
    assert!(t.block(&U1Irrep::new(0)).is_ok());
}

#[cfg(feature = "racah-generated")]
mod su3 {
    use std::sync::Arc;

    use super::*;
    use tenet::typed::SUNFusionRule;

    pub(super) fn legs() -> [GradedSpace<SUNFusionRule>; 3] {
        let provider = Arc::new(SUNFusionRule::new(3).unwrap());
        let octet = GradedSpace::try_new_with_arc(
            Arc::clone(&provider),
            [(vec![1i64, 1], 2), (vec![0, 0], 1)],
        )
        .unwrap();
        let chiral =
            GradedSpace::try_new_with_arc(provider, [(vec![1i64, 0], 2), (vec![0, 1], 1)]).unwrap();
        let dual = chiral.try_dual().unwrap();
        [octet, chiral, dual]
    }

    fn run<D: Val>(what: &str) {
        let rt = Runtime::builder().build().unwrap();
        let [octet, chiral, dual] = legs();
        let a = label_case::<_, D>(what, &rt, &[&octet, &octet], &[&octet]);
        assert!(
            (0..a.subblock_count()).any(|i| a
                .subblock_fusion_trees(i)
                .unwrap()
                .codomain_vertices()[0]
                .get()
                == 2),
            "{what}: fixture must carry outer multiplicity"
        );
        let b = label_case::<_, D>(what, &rt, &[&octet], &[&chiral, &dual]);
        compose_case(what, &a, &b);
        adjoint_view_case(what, &b, &b.adjoint().unwrap());

        let values = |n: usize| (0..n).map(|i| D::make(i as u64 + 11)).collect::<Vec<_>>();
        let d = TensorMap::<_, D>::diagonal(
            &rt,
            &octet,
            [
                SectorSpectrum {
                    sector: vec![1i64, 1],
                    values: values(2),
                },
                SectorSpectrum {
                    sector: vec![0, 0],
                    values: values(1),
                },
            ],
        )
        .unwrap();
        for (sector, n) in [(vec![1i64, 1], 2), (vec![0, 0], 1)] {
            let block = d.block(&sector).unwrap();
            let CoupledBlockPayload::Diagonal(stored) = block.payload() else {
                panic!("{what}: a compact diagonal block must borrow its spectrum");
            };
            assert_eq!(stored, values(n).as_slice(), "{what}: diagonal {sector:?}");
            assert_eq!(
                block.get(0, n - 1).unwrap().c(),
                if n == 1 {
                    values(1)[0].c()
                } else {
                    Complex64::new(0.0, 0.0)
                }
            );
        }
    }

    #[test]
    fn checked_generic_su3_blocks_match_label_adjoint_and_composition_oracles() {
        run::<f64>("SU(3) f64");
        run::<Complex64>("SU(3) c64");
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use tenet::typed::CudaStorage;

    /// The device view has the Host view's geometry, and the device region it
    /// names holds the Host block's entries.
    fn device_case<R, D>(what: &str, host: &TensorMap<R, D>)
    where
        R: TypedSectorAdmission,
        R::Mode: tenet::typed::TypedTensorModeDispatch<R>,
        D: Val + tenet::typed::CudaPayload + tenet::dense::CudaScalar,
    {
        let ctx = tenet::dense::CudaDenseContext::new(0).unwrap();
        let device: TensorMap<R, D, CudaStorage<D>> = host.to_cuda().unwrap();
        let host_blocks: Vec<_> = host.blocks().unwrap().collect();
        let device_blocks: Vec<_> = device.blocks().unwrap().collect();
        assert_eq!(host_blocks.len(), device_blocks.len(), "{what}");
        let mut downloaded = None;
        for ((sector, h), (device_sector, d)) in host_blocks.iter().zip(&device_blocks) {
            assert_eq!(sector, device_sector, "{what}");
            assert_eq!((h.rows(), h.cols()), (d.rows(), d.cols()), "{what}");
            let (
                CoupledBlockPayload::Dense {
                    offset: host_offset,
                    adjoint: host_adjoint,
                    ..
                },
                CoupledBlockPayload::Dense {
                    storage,
                    offset,
                    adjoint,
                },
            ) = (h.payload(), d.payload())
            else {
                panic!("{what}: dense views expected");
            };
            assert_eq!((host_offset, host_adjoint), (offset, adjoint), "{what}");
            let data: &Vec<D> = downloaded.get_or_insert_with(|| storage.download(&ctx).unwrap());
            let want = read(h);
            for c in 0..d.cols() {
                for r in 0..d.rows() {
                    let value = if adjoint {
                        data[offset + c + r * d.cols()].c().conj()
                    } else {
                        data[offset + r + c * d.rows()].c()
                    };
                    assert_eq!(value, want.at(r, c), "{what}: ({r}, {c}) of {sector:?}");
                }
            }
        }
    }

    fn run<R, D>(what: &str, rt: &Runtime, v: &GradedSpace<R>, w: &GradedSpace<R>)
    where
        R: TypedSectorAdmission + tenet::core::MultiplicityFreeRigidSymbols<Scalar = f64>,
        R::Mode: tenet::typed::TypedTensorModeDispatch<R>
            + tenet::typed::TypedTensorConstructionDispatch<R, D>
            + tenet::typed::TypedTensorAdjointDispatch<R, D>,
        D: Val + tenet::typed::CudaPayload + tenet::dense::CudaScalar,
    {
        let t = label_case::<R, D>(what, rt, &[v, w], &[v]);
        device_case(what, &t);
        device_case(&format!("{what} adjoint"), &t.adjoint().unwrap());
    }

    #[test]
    #[ignore]
    fn device_block_views_address_the_host_block_entries() {
        let rt = Runtime::builder().cuda(0).build().unwrap();
        let (v, w) = u1_legs();
        run::<_, f64>("U(1) f64", &rt, &v, &w);
        run::<_, Complex64>("U(1) c64", &rt, &v, &w);
        let (v, w) = su2_legs();
        run::<_, f64>("SU(2) f64", &rt, &v, &w);
        run::<_, Complex64>("SU(2) c64", &rt, &v, &w);
        let (v, w) = fz2u1_legs();
        run::<_, f64>("fZ2xU(1) f64", &rt, &v, &w);
        run::<_, Complex64>("fZ2xU(1) c64", &rt, &v, &w);
    }

    #[cfg(feature = "racah-generated")]
    #[test]
    #[ignore]
    fn checked_generic_su3_device_block_views_address_the_host_block_entries() {
        let rt = Runtime::builder().cuda(0).build().unwrap();
        let [octet, _, _] = super::su3::legs();
        let t = label_case::<_, f64>("SU(3) f64", &rt, &[&octet, &octet], &[&octet]);
        device_case("SU(3) f64", &t);
        device_case("SU(3) f64 adjoint", &t.adjoint().unwrap());
        let t = label_case::<_, Complex64>("SU(3) c64", &rt, &[&octet, &octet], &[&octet]);
        device_case("SU(3) c64", &t);
        device_case("SU(3) c64 adjoint", &t.adjoint().unwrap());
    }
}
