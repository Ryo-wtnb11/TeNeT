//! Multiplicity-free public linear-algebra conformance (#1002).
//!
//! These are deliberately public-API tests. They cover the residual provider
//! fixtures in #1002, not every constructible `FusionRule` implementation.

use std::sync::Arc;

use tenet::core::{
    product_sector, FermionParityFusionRule, MultiplicityFreeRigidSymbols, PackedProductCodec,
    ProductFusionRule, ProductSectorLayout, SU2FusionRule, SU2Irrep, SectorCodec, Su2SectorLayout,
    U1FusionRule, U1Irrep, U1SectorLayout, Z2Irrep, ZNFusionRule,
};
use tenet::prelude::{GradedSpace, Runtime, TensorMap, Truncation};

type Fz2SectorLayout = tenet::core::Fz2SectorLayout;
type Fz2U1Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
type Fz2U1Layout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
type Fz2U1Su2Codec = PackedProductCodec<Fz2U1Layout, Su2SectorLayout>;
type Fz2U1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule, Fz2U1Codec>;
type Fz2U1Su2Rule = ProductFusionRule<Fz2U1Rule, SU2FusionRule, Fz2U1Su2Codec>;

fn runtime() -> Runtime {
    Runtime::builder().dense_threads(1).build().unwrap()
}

macro_rules! assert_close {
    ($actual:expr, $expected:expr) => {{
        let actual = $actual;
        let expected = $expected;
        let error = actual.add(expected, 1.0, -1.0).unwrap().norm().unwrap();
        assert!(
            error <= 1e-9 * (1.0 + expected.norm().unwrap()),
            "residual {error}"
        );
    }};
}

macro_rules! assert_complex_close {
    ($actual:expr, $expected:expr) => {{
        let actual = $actual;
        let expected = $expected;
        assert_eq!(actual.data().len(), expected.data().len());
        for (actual, expected) in actual.data().iter().zip(expected.data()) {
            assert!(
                (*actual - *expected).norm() <= 1e-9 * (1.0 + expected.norm()),
                "{actual} != {expected}"
            );
        }
    }};
}

macro_rules! assert_provider {
    ($provider:expr; $($tensor:expr),+ $(,)?) => {
        $(assert!(std::ptr::eq($tensor.provider(), $provider.as_ref()));)+
    };
}

/// The rectangular rows deliberately use a rank-two diagonal embedded in a
/// larger block.  This makes both null directions and polar isometry laws
/// observable without pinning a dense-factor gauge.
macro_rules! factor_conformance {
    ($name:expr, $rule:expr, $pairs:expr) => {{
        let rt = runtime();
        let provider = Arc::new($rule);
        let pairs = $pairs;
        let endo_space = GradedSpace::try_new(Arc::clone(&provider), pairs.clone(), false).unwrap();
        let tall_pairs: Vec<_> = pairs
            .iter()
            .map(|(sector, degeneracy)| (sector.clone(), degeneracy + 1))
            .collect();
        let wide_pairs: Vec<_> = pairs
            .iter()
            .map(|(sector, degeneracy)| (sector.clone(), *degeneracy))
            .collect();
        let tall_space = GradedSpace::try_new(Arc::clone(&provider), tall_pairs, false).unwrap();
        let wide_space = GradedSpace::try_new(Arc::clone(&provider), wide_pairs, false).unwrap();
        let tall: TensorMap<_, f64> =
            TensorMap::from_block_fn(&rt, [&tall_space], [&wide_space], |_, index| {
                if index[0] == index[1] {
                    [4.0, 2.0][index[0]]
                } else {
                    0.0
                }
            })
            .unwrap();
        let wide: TensorMap<_, f64> =
            TensorMap::from_block_fn(&rt, [&wide_space], [&tall_space], |_, index| {
                if index[0] == index[1] {
                    [4.0, 2.0][index[0]]
                } else {
                    0.0
                }
            })
            .unwrap();

        let (u, s, vh) = tall.svd_compact().unwrap();
        assert_provider!(provider; u, s, vh);
        assert_close!(&u.compose(&s).unwrap().compose(&vh).unwrap(), &tall);
        let (u, s, vh) = tall.svd_full().unwrap();
        assert_provider!(provider; u, s, vh);
        assert_close!(&u.compose(&s).unwrap().compose(&vh).unwrap(), &tall);
        let trunc = tall.svd_trunc(&Truncation::rank(1)).unwrap();
        assert_provider!(provider; trunc.u, trunc.s, trunc.vh);
        let reconstructed = trunc
            .u
            .compose(&trunc.s)
            .unwrap()
            .compose(&trunc.vh)
            .unwrap();
        let error = reconstructed.add(&tall, 1.0, -1.0).unwrap().norm().unwrap();
        assert!((error - trunc.error).abs() <= 1e-9 * (1.0 + trunc.error));
        assert!(trunc.error > 0.0, $name);
        let singular_values = tall.svd_vals().unwrap();
        assert!(singular_values
            .iter()
            .all(|entry| entry.values == [4.0, 2.0]));
        let weighted_norm_squared: f64 = singular_values
            .iter()
            .map(|entry| {
                provider.dim_scalar(
                    SectorCodec::encode_sector(provider.as_ref(), &entry.sector).unwrap(),
                )
                    * entry.values.iter().map(|value| value * value).sum::<f64>()
            })
            .sum();
        assert!((tall.norm().unwrap().powi(2) - weighted_norm_squared).abs() <= 1e-9);

        let (q, r) = tall.qr_compact().unwrap();
        assert_provider!(provider; q, r);
        assert_close!(&q.compose(&r).unwrap(), &tall);
        let id = TensorMap::id(&rt, q.domain().iter()).unwrap();
        assert_close!(&q.adjoint().unwrap().compose(&q).unwrap(), &id);
        let (q, r) = tall.left_orth().unwrap();
        assert_provider!(provider; q, r);
        assert_close!(&q.compose(&r).unwrap(), &tall);
        let (q, r) = tall.qr_full().unwrap();
        assert_provider!(provider; q, r);
        assert_close!(&q.compose(&r).unwrap(), &tall);
        let id = TensorMap::id(&rt, q.domain().iter()).unwrap();
        assert_close!(&q.adjoint().unwrap().compose(&q).unwrap(), &id);

        let (l, q) = wide.lq_compact().unwrap();
        assert_provider!(provider; l, q);
        assert_close!(&l.compose(&q).unwrap(), &wide);
        let id = TensorMap::id(&rt, q.codomain().iter()).unwrap();
        assert_close!(&q.compose(&q.adjoint().unwrap()).unwrap(), &id);
        let (l, q) = wide.right_orth().unwrap();
        assert_provider!(provider; l, q);
        assert_close!(&l.compose(&q).unwrap(), &wide);
        let (l, q) = wide.lq_full().unwrap();
        assert_provider!(provider; l, q);
        assert_close!(&l.compose(&q).unwrap(), &wide);
        let id = TensorMap::id(&rt, q.codomain().iter()).unwrap();
        assert_close!(&q.compose(&q.adjoint().unwrap()).unwrap(), &id);

        let left = tall.left_null().unwrap();
        assert_provider!(provider; left);
        assert!(
            left.adjoint()
                .unwrap()
                .compose(&tall)
                .unwrap()
                .norm()
                .unwrap()
                <= 1e-9,
            $name
        );
        let left_id = TensorMap::id(&rt, left.domain().iter()).unwrap();
        assert_close!(&left.adjoint().unwrap().compose(&left).unwrap(), &left_id);
        let right = wide.right_null().unwrap();
        assert_provider!(provider; right);
        assert!(
            wide.compose(&right.adjoint().unwrap())
                .unwrap()
                .norm()
                .unwrap()
                <= 1e-9,
            $name
        );
        let right_id = TensorMap::id(&rt, right.codomain().iter()).unwrap();
        assert_close!(
            &right.compose(&right.adjoint().unwrap()).unwrap(),
            &right_id
        );

        let (w, p) = tall.left_polar().unwrap();
        assert_provider!(provider; w, p);
        assert_close!(&w.compose(&p).unwrap(), &tall);
        let id = TensorMap::id(&rt, w.domain().iter()).unwrap();
        assert_close!(&w.adjoint().unwrap().compose(&w).unwrap(), &id);
        assert!(p.is_hermitian(1e-10).unwrap());
        assert!(p
            .eigh_vals()
            .unwrap()
            .iter()
            .flat_map(|s| &s.values)
            .all(|&x| x >= -1e-10));
        let (p, w) = wide.right_polar().unwrap();
        assert_provider!(provider; p, w);
        assert_close!(&p.compose(&w).unwrap(), &wide);
        let id = TensorMap::id(&rt, w.codomain().iter()).unwrap();
        assert_close!(&w.compose(&w.adjoint().unwrap()).unwrap(), &id);
        assert!(p.is_hermitian(1e-10).unwrap());
        assert!(p
            .eigh_vals()
            .unwrap()
            .iter()
            .flat_map(|s| &s.values)
            .all(|&x| x >= -1e-10));

        let h: TensorMap<_, f64> =
            TensorMap::from_block_fn(&rt, [&endo_space], [&endo_space], |_, index| {
                [[3.0, 1.0], [1.0, 3.0]][index[0]][index[1]]
            })
            .unwrap();
        let (d, v) = h.eigh_full().unwrap();
        assert_provider!(provider; d, v);
        assert_close!(
            &v.compose(&d)
                .unwrap()
                .compose(&v.adjoint().unwrap())
                .unwrap(),
            &h
        );
        assert!(h
            .eigh_vals()
            .unwrap()
            .iter()
            .all(|entry| entry.values == [4.0, 2.0]));
        let trunc = h.eigh_trunc(&Truncation::rank(1)).unwrap();
        assert_provider!(provider; trunc.d, trunc.v);
        let reconstructed = trunc
            .v
            .compose(&trunc.d)
            .unwrap()
            .compose(&trunc.v.adjoint().unwrap())
            .unwrap();
        let error = reconstructed.add(&h, 1.0, -1.0).unwrap().norm().unwrap();
        assert!((error - trunc.error).abs() <= 1e-9 * (1.0 + trunc.error));
        assert!(trunc.error > 0.0, $name);

        let g: TensorMap<_, f64> =
            TensorMap::from_block_fn(&rt, [&endo_space], [&endo_space], |_, index| {
                [[3.0, 1.0], [0.0, 1.0]][index[0]][index[1]]
            })
            .unwrap();
        let (d, v) = g.eig_full().unwrap();
        assert_provider!(provider; d, v);
        assert_complex_close!(
            &v.compose(&d).unwrap().compose(&v.inv().unwrap()).unwrap(),
            &g.to_c64()
        );
        assert!(g
            .eig_vals()
            .unwrap()
            .iter()
            .all(|entry| entry.values == [3.0.into(), 1.0.into()]));
        let trunc = g.eig_trunc(&Truncation::rank(1)).unwrap();
        assert_provider!(provider; trunc.d, trunc.v);
        assert_complex_close!(
            &g.to_c64().compose(&trunc.v).unwrap(),
            &trunc.v.compose(&trunc.d).unwrap()
        );
        assert!(trunc.error > 0.0, $name);

        let id = TensorMap::id(&rt, h.domain().iter()).unwrap();
        let inverse = h.inv().unwrap();
        assert_provider!(provider; inverse);
        assert_close!(&h.compose(&inverse).unwrap(), &id);
        let exponential = h.exp().unwrap();
        assert_provider!(provider; exponential);
        assert_close!(
            &h.scale(-1.0)
                .exp()
                .unwrap()
                .compose(&exponential)
                .unwrap(),
            &id
        );
        let squared = h.powi(2).unwrap();
        assert_provider!(provider; squared);
        assert_close!(&squared, &h.compose(&h).unwrap());
        let solved = h.solve(&id).unwrap();
        assert_provider!(provider; solved);
        assert_close!(&solved, &inverse);
        let solved_right = id.solve_right(&h).unwrap();
        assert_provider!(provider; solved_right);
        assert_close!(&solved_right, &inverse);
        let pseudo = tall.pinv(1e-12).unwrap();
        assert_provider!(provider; pseudo);
        assert_close!(
            &tall.compose(&pseudo).unwrap().compose(&tall).unwrap(),
            &tall
        );
        let aa_plus = tall.compose(&pseudo).unwrap();
        let a_plus_a = pseudo.compose(&tall).unwrap();
        assert_close!(&a_plus_a.compose(&pseudo).unwrap(), &pseudo);
        assert_close!(&aa_plus.adjoint().unwrap(), &aa_plus);
        assert_close!(&a_plus_a.adjoint().unwrap(), &a_plus_a);
        let left_projector = left.compose(&left.adjoint().unwrap()).unwrap();
        let id = TensorMap::id(&rt, tall.codomain().iter()).unwrap();
        assert_close!(&left_projector.add(&aa_plus, 1.0, 1.0).unwrap(), &id);
        let diagonal: TensorMap<_, f64> =
            TensorMap::from_block_fn(&rt, [&endo_space], [&endo_space], |_, index| {
                if index[0] == index[1] {
                    [4.0, 9.0][index[0]]
                } else {
                    0.0
                }
            })
            .unwrap();
        let root = diagonal.sqrt().unwrap();
        assert_provider!(provider; root);
        assert!(root.data().iter().any(|&value| value == 2.0));
        assert!(root.data().iter().any(|&value| value == 3.0));
        assert!(root
            .data()
            .iter()
            .all(|&value| value == 0.0 || value == 2.0 || value == 3.0));
        assert_close!(&root.compose(&root).unwrap(), &diagonal);
        let before = tall.data().to_vec();
        assert!(tall.pinv(-1.0).is_err());
        assert_eq!(tall.data(), before.as_slice());
    }};
}

#[test]
fn public_multiplicity_free_linalg_conformance() {
    factor_conformance!(
        "U(1)",
        U1FusionRule,
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 2)]
    );
    let z3 = ZNFusionRule::new(3).unwrap();
    let z3_pairs = [(z3.irrep(0), 2), (z3.irrep(1), 2)];
    factor_conformance!("Z3", z3, z3_pairs);
    factor_conformance!(
        "CU1 vacuum/pseudo/charged",
        tenet::core::CU1FusionRule,
        [
            (tenet::core::CU1Irrep::VACUUM, 2),
            (tenet::core::CU1Irrep::PSEUDOSCALAR, 2),
            (tenet::core::CU1Irrep::from_twice_charge(1), 2),
        ]
    );
    factor_conformance!("fZ2 odd", FermionParityFusionRule, [(Z2Irrep::ODD, 2)]);
    factor_conformance!(
        "SU2 nonzero spin",
        SU2FusionRule,
        [(SU2Irrep::from_twice_spin(1), 2)]
    );
    factor_conformance!(
        "fZ2xU1",
        Fz2U1Rule::new(FermionParityFusionRule, U1FusionRule),
        [
            (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 2),
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
        ]
    );
    factor_conformance!(
        "fZ2xU1xSU2",
        Fz2U1Su2Rule::new(
            Fz2U1Rule::new(FermionParityFusionRule, U1FusionRule),
            SU2FusionRule,
        ),
        [(
            product_sector(
                product_sector(Z2Irrep::ODD, U1Irrep::new(1)),
                SU2Irrep::from_twice_spin(1),
            ),
            2,
        )]
    );
}
