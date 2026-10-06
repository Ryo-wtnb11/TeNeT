/// Every factorization of `$t` (an endomorphism) as `(family, error text)`,
/// for inputs that must all be refused by the shared finite-input stage
/// (#1986). The error is read through `Debug`, which both fusion modes'
/// facade errors implement.
macro_rules! nonfinite_factor_errors {
    ($t:expr) => {{
        let t = $t;
        let rows: &[usize] = &[0];
        let cols: &[usize] = &[1];
        let text = |result: Result<(), _>| format!("{:?}", result.unwrap_err());
        vec![
            ("svd", text(t.svd_vals(rows, cols).map(drop))),
            ("svd", text(t.svd_compact(rows, cols).map(drop))),
            ("svd", text(t.svd_full(rows, cols).map(drop))),
            ("qr", text(t.qr_compact(rows, cols).map(drop))),
            ("qr", text(t.qr_full(rows, cols).map(drop))),
            ("lq", text(t.lq_compact(rows, cols).map(drop))),
            ("lq", text(t.lq_full(rows, cols).map(drop))),
            ("null", text(t.left_null(rows, cols).map(drop))),
            ("null", text(t.right_null(rows, cols).map(drop))),
            ("polar", text(t.left_polar(rows, cols).map(drop))),
            ("polar", text(t.right_polar(rows, cols).map(drop))),
            (
                "eigh",
                text(
                    t.eigh_vals(rows, cols, tenet::typed::HermitianTol::DEFAULT)
                        .map(drop),
                ),
            ),
            (
                "eigh",
                text(
                    t.eigh_full(rows, cols, tenet::typed::HermitianTol::DEFAULT)
                        .map(drop),
                ),
            ),
            ("eig", text(t.eig_vals(rows, cols).map(drop))),
            ("eig", text(t.eig_full(rows, cols).map(drop))),
        ]
    }};
}

/// Asserts that every entry of [`nonfinite_factor_errors`] names its
/// family's finite-input refusal.
pub(crate) fn assert_finite_input_refusals(errors: Vec<(&str, String)>, case: &str) {
    for (family, error) in errors {
        let expected = format!("{family} input components must be finite");
        assert!(
            error.contains("InvalidArgument") && error.contains(&expected),
            "{case}: expected {expected:?}, got {error}"
        );
    }
}
