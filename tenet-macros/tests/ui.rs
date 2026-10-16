//! Every rejected `contract!` form is a compile error with its hint.

#[test]
fn rejected_forms() {
    trybuild::TestCases::new().compile_fail("tests/ui/*.rs");
}
