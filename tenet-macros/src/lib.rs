//! `tensor!` — @tensor-style index-notation contraction for TeNeT.
//!
//! Expression form (returns `Result<Tensor, Error>`):
//!
//! ```text
//! let c = tensor!([a, b; g, h] = x[a, b; i, j] * y[i, j; g, h])?;
//! let n = tensor!([] = conj(psi)[p; l, r] * psi[p; l, r])?;   // rank-0
//! ```
//!
//! - The output signature comes first: `[codomain; domain]`; `;` optional
//!   (`[a, b]` = all-codomain output, `[]` = scalar / rank-0 output).
//! - RHS terms are `expr[labels]` products; `expr` is an identifier, a
//!   field-access chain (`svd.u`, `pair.0`), a parenthesized expression, or
//!   `conj(expr)` marking an adjoint operand. Each `expr` must evaluate to
//!   a `Tensor` or `&Tensor`.
//! - A label appearing on two operands is contracted; a label appearing
//!   twice on ONE operand is a partial trace of that operand (TensorKit
//!   `@tensor a[i, i; j]`); a label appearing once must be listed in the
//!   output. Violations are compile errors.
//! - Lowers to `tenet_network::contract_network` (planner IR directly; no
//!   einsum strings).
//!
//! **Fermionic semantics**: `tensor!` follows TensorKit `@tensor` /
//! `tensorcontract!` — dual contracted legs are twisted with the fermionic
//! supertrace twist. `Tensor::compose` / `&a * &b` (TensorKit `A * B` /
//! `mul!`) never twist. Bosonic rules are identical either way; fermionic
//! rules can differ by signs — see the worked example on
//! `Tensor::compose`.

use proc_macro::TokenStream;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{parenthesized, Expr, Ident, Token};

/// One `[labels]` group: flat label list plus the `;` position, if any.
struct LabelGroup {
    labels: Vec<String>,
    split: Option<usize>,
}

impl Parse for LabelGroup {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let content;
        syn::bracketed!(content in input);
        let mut labels = Vec::new();
        let mut split = None;
        while !content.is_empty() {
            if content.peek(Token![;]) {
                content.parse::<Token![;]>()?;
                if split.is_some() {
                    return Err(content.error("more than one `;` in a label group"));
                }
                split = Some(labels.len());
                continue;
            }
            let label: Ident = content.parse()?;
            labels.push(label.to_string());
            if content.peek(Token![,]) {
                content.parse::<Token![,]>()?;
            }
        }
        Ok(Self { labels, split })
    }
}

/// One RHS operand: `expr[labels]`, `(expr)[labels]` or `conj(expr)[labels]`,
/// where a bare `expr` is an identifier optionally followed by a
/// field-access chain (`svd.u[a; b]`, `pair.0[i, j]`).
struct Operand {
    tensor: Expr,
    conj: bool,
    group: LabelGroup,
}

/// Parses `ident (. member)*` up to the operand's `[labels]` bracket,
/// building the field-access chain (`svd.u`, `x.0.1` is out of scope: a
/// float-literal chain — parenthesize instead).
fn parse_field_chain(input: ParseStream) -> syn::Result<Expr> {
    let ident: Ident = input.parse()?;
    let mut expr: Expr = syn::parse_quote!(#ident);
    while input.peek(Token![.]) {
        input.parse::<Token![.]>()?;
        let member: syn::Member = input.parse()?;
        expr = Expr::Field(syn::ExprField {
            attrs: Vec::new(),
            base: Box::new(expr),
            dot_token: Default::default(),
            member,
        });
    }
    Ok(expr)
}

impl Parse for Operand {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let (tensor, conj) = if input.peek(Ident) && input.peek2(syn::token::Paren) {
            let ident: Ident = input.parse()?;
            if ident != "conj" {
                return Err(syn::Error::new(
                    ident.span(),
                    "expected `conj(...)`, an identifier or field access, or a \
                     parenthesized expression",
                ));
            }
            let content;
            parenthesized!(content in input);
            (content.parse::<Expr>()?, true)
        } else if input.peek(syn::token::Paren) {
            let content;
            parenthesized!(content in input);
            (content.parse::<Expr>()?, false)
        } else {
            (parse_field_chain(input)?, false)
        };
        let group: LabelGroup = input.parse()?;
        Ok(Self {
            tensor,
            conj,
            group,
        })
    }
}

/// `[out labels] = operand * operand * ...`
struct TensorExpr {
    output: LabelGroup,
    operands: Vec<Operand>,
}

impl Parse for TensorExpr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let output: LabelGroup = input.parse()?;
        input.parse::<Token![=]>()?;
        let operands: Punctuated<Operand, Token![*]> = Punctuated::parse_separated_nonempty(input)?;
        Ok(Self {
            output,
            operands: operands.into_iter().collect(),
        })
    }
}

/// Compile-time label validation, following TensorOperations `@tensor`
/// semantics: every RHS label appears either once (open: must be listed in
/// the output) or exactly twice (contracted: on two different operands it
/// is a pairwise contraction, twice on the SAME operand it is a partial
/// trace of that operand; either way it must not be in the output). Three
/// or more appearances (hyperedges) are rejected, and output labels are
/// unique and come from the RHS.
fn check_labels(inputs: &[Vec<String>], output: &[String]) -> Result<(), String> {
    let mut seen_output = std::collections::BTreeSet::new();
    for label in output {
        if !seen_output.insert(label) {
            return Err(format!("duplicate output label `{label}`"));
        }
    }
    // label -> (total occurrences, distinct operands)
    let mut counts: std::collections::BTreeMap<&str, (usize, usize)> = Default::default();
    for labels in inputs {
        let mut seen_here = std::collections::BTreeSet::new();
        for label in labels {
            let entry = counts.entry(label.as_str()).or_insert((0, 0));
            entry.0 += 1;
            if seen_here.insert(label) {
                entry.1 += 1;
            }
        }
    }
    for label in output {
        if !counts.contains_key(label.as_str()) {
            return Err(format!(
                "output label `{label}` does not appear on any tensor"
            ));
        }
    }
    for (label, (total, operands)) in &counts {
        let in_output = output.iter().any(|l| l == label);
        if *operands > 2 {
            return Err(format!(
                "label `{label}` appears on {operands} tensors (more than two)"
            ));
        }
        if in_output {
            if *total != 1 {
                return Err(format!(
                    "output label `{label}` appears {total} times on the right-hand \
                     side (an output label must appear exactly once)"
                ));
            }
        } else if *total != 2 {
            return Err(format!(
                "label `{label}` appears {total} time(s) but is not an output \
                 label (a contracted or traced label must appear exactly twice)"
            ));
        }
    }
    Ok(())
}

/// @tensor-style contraction over TeNeT user-layer tensors; see the crate
/// docs for the syntax. Evaluates to `Result<Tensor, tenet::prelude::Error>`.
#[proc_macro]
pub fn tensor(input: TokenStream) -> TokenStream {
    let parsed = syn::parse_macro_input!(input as TensorExpr);

    let inputs: Vec<Vec<String>> = parsed
        .operands
        .iter()
        .map(|op| op.group.labels.clone())
        .collect();
    if let Err(message) = check_labels(&inputs, &parsed.output.labels) {
        return syn::Error::new(proc_macro2::Span::call_site(), message)
            .to_compile_error()
            .into();
    }

    let tensors = parsed.operands.iter().map(|op| {
        let tensor = &op.tensor;
        quote! { &#tensor }
    });
    let labels = parsed.operands.iter().map(|op| &op.group.labels);
    let conj = parsed.operands.iter().map(|op| op.conj);
    let splits = parsed
        .operands
        .iter()
        .map(|op| option_tokens(op.group.split));
    let output = &parsed.output.labels;
    let out_split = option_tokens(parsed.output.split);

    quote! {
        {
            const __TENET_TOPOLOGY: ::tenet_network::StaticTopologySpec =
                ::tenet_network::StaticTopologySpec {
                    inputs: &[#(&[#(#labels),*]),*],
                    conj: &[#(#conj),*],
                    codomain_splits: &[#(#splits),*],
                    output: &[#(#output),*],
                    output_codomain_rank: #out_split,
                };
            ::tenet_network::contract_static_network(
                &[#(#tensors),*],
                &__TENET_TOPOLOGY,
            )
        }
    }
    .into()
}

fn option_tokens(value: Option<usize>) -> proc_macro2::TokenStream {
    match value {
        Some(v) => quote!(::core::option::Option::Some(#v)),
        None => quote!(::core::option::Option::None),
    }
}

#[cfg(test)]
mod tests {
    use super::{check_labels, Operand};
    use quote::ToTokens;

    fn parse_operand(source: &str) -> syn::Result<Operand> {
        syn::parse_str::<Operand>(source)
    }

    #[test]
    fn operand_accepts_bare_identifier() {
        let op = parse_operand("a[i, j]").unwrap();
        assert!(!op.conj);
        assert_eq!(op.tensor.to_token_stream().to_string(), "a");
        assert_eq!(op.group.labels, ["i", "j"]);
    }

    #[test]
    fn operand_accepts_field_access_chain() {
        let op = parse_operand("svd.u[a; b]").unwrap();
        assert!(!op.conj);
        assert_eq!(op.tensor.to_token_stream().to_string(), "svd . u");
        assert_eq!(op.group.labels, ["a", "b"]);
        assert_eq!(op.group.split, Some(1));

        let op = parse_operand("net.site.left[i]").unwrap();
        assert_eq!(op.tensor.to_token_stream().to_string(), "net . site . left");
    }

    #[test]
    fn operand_accepts_tuple_index_field() {
        let op = parse_operand("pair.0[i, j]").unwrap();
        assert_eq!(op.tensor.to_token_stream().to_string(), "pair . 0");
    }

    #[test]
    fn operand_conj_and_parens_still_parse() {
        let op = parse_operand("conj(svd.u)[a; b]").unwrap();
        assert!(op.conj);
        assert_eq!(op.tensor.to_token_stream().to_string(), "svd . u");

        let op = parse_operand("(f(x))[i]").unwrap();
        assert!(!op.conj);
        assert_eq!(op.tensor.to_token_stream().to_string(), "f (x)");
    }

    #[test]
    fn operand_rejects_non_conj_call() {
        assert!(parse_operand("foo(x)[i]").is_err());
    }

    fn labels(groups: &[&[&str]]) -> Vec<Vec<String>> {
        groups
            .iter()
            .map(|g| g.iter().map(|s| s.to_string()).collect())
            .collect()
    }

    fn out(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn valid_pairwise_and_scalar_networks_pass() {
        check_labels(&labels(&[&["a", "b"], &["b", "c"]]), &out(&["a", "c"])).unwrap();
        check_labels(&labels(&[&["a", "b"], &["a", "b"]]), &out(&[])).unwrap();
        // 3-tensor chain with conj-style shared labels.
        check_labels(
            &labels(&[&["p", "l", "r"], &["p", "q"], &["q", "l", "r"]]),
            &out(&[]),
        )
        .unwrap();
        // Single-tensor relabel/permute.
        check_labels(&labels(&[&["a", "b"]]), &out(&["b", "a"])).unwrap();
    }

    #[test]
    fn duplicate_output_label_is_rejected() {
        let err = check_labels(&labels(&[&["a", "b"]]), &out(&["a", "a"])).unwrap_err();
        assert!(err.contains("duplicate output label `a`"), "{err}");
    }

    #[test]
    fn output_label_missing_from_rhs_is_rejected() {
        let err =
            check_labels(&labels(&[&["a", "b"], &["b", "c"]]), &out(&["a", "z"])).unwrap_err();
        assert!(err.contains("output label `z`"), "{err}");
    }

    #[test]
    fn dangling_contracted_label_is_rejected() {
        // `c` appears once and is not an output label.
        let err = check_labels(&labels(&[&["a", "b"], &["b", "c"]]), &out(&["a"])).unwrap_err();
        assert!(err.contains("label `c` appears 1 time(s)"), "{err}");
    }

    #[test]
    fn hyperedge_label_is_rejected() {
        let err = check_labels(
            &labels(&[&["a", "b"], &["b", "c"], &["b", "d"]]),
            &out(&["a", "c", "d"]),
        )
        .unwrap_err();
        assert!(err.contains("label `b` appears on 3 tensors"), "{err}");
    }

    #[test]
    fn batch_output_label_is_rejected() {
        // `a` is shared by both operands AND requested in the output.
        let err = check_labels(&labels(&[&["a", "b"], &["a", "b"]]), &out(&["a"])).unwrap_err();
        assert!(err.contains("output label `a` appears 2 times"), "{err}");
    }

    #[test]
    fn trace_pairs_within_one_operand_are_accepted() {
        // Full trace to a scalar.
        check_labels(&labels(&[&["a", "a"]]), &out(&[])).unwrap();
        // Partial trace with an open leg.
        check_labels(&labels(&[&["a", "a", "j"]]), &out(&["j"])).unwrap();
        // Trace combined with a pairwise contraction.
        check_labels(&labels(&[&["a", "a", "j"], &["j", "m"]]), &out(&["m"])).unwrap();
    }

    #[test]
    fn traced_label_in_output_is_rejected() {
        let err = check_labels(&labels(&[&["a", "a"]]), &out(&["a"])).unwrap_err();
        assert!(err.contains("output label `a` appears 2 times"), "{err}");
    }

    #[test]
    fn trace_pair_plus_third_occurrence_is_rejected() {
        // `a` twice on operand 0 and once on operand 1: three appearances.
        let err = check_labels(&labels(&[&["a", "a"], &["a", "b"]]), &out(&["b"])).unwrap_err();
        assert!(err.contains("label `a` appears 3 time(s)"), "{err}");
    }

    #[test]
    fn label_thrice_on_one_operand_is_rejected() {
        let err = check_labels(&labels(&[&["a", "a", "a"]]), &out(&[])).unwrap_err();
        assert!(err.contains("label `a` appears 3 time(s)"), "{err}");
    }
}
