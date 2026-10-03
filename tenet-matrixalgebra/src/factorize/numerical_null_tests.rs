use super::null_space::numerical_rank_and_compact_basis;
use super::*;
use crate::tests::scripted_executor::{Action, Observer, Op, Script, ScriptedExecutor};

/// Owned SVD only (`svd_into` fails); records the returned `U` buffer.
#[derive(Default)]
struct OwnedSvdSpy {
    u_pointer: Option<usize>,
}

impl Observer for OwnedSvdSpy {
    fn script(script: &mut Script) {
        script
            .set_all(&[Op::Svd, Op::Qr, Op::Eigh], Action::Forward)
            .fail(
                &[Op::SvdInto],
                None,
                "svd_into",
                "numerical null must consume owned SVD factors",
            );
    }

    fn outputs(&mut self, op: Op, outputs: &mut Vec<DenseTensor>) {
        if op == Op::Svd {
            self.u_pointer = Some(
                outputs[0]
                    .as_f64_slice()
                    .expect("f64 null fixture must return f64 U")
                    .as_ptr() as usize,
            );
        }
    }
}

#[test]
fn numerical_null_left_basis_keeps_owned_svd_u() {
    let mut dense = ScriptedExecutor::<OwnedSvdSpy>::default();
    let (rank, u) = numerical_rank_and_compact_basis(
        &mut dense,
        &[1.0_f64, 0.0, 0.0, 0.0, 2.0, 0.0],
        3,
        2,
        FactorSide::Left,
    )
    .unwrap();

    assert_eq!(rank, 2);
    assert_eq!(u.len(), 6);
    assert_eq!(dense.counts().svd_into, 0);
    assert_eq!(u.as_ptr() as usize, dense.u_pointer.unwrap());
}
