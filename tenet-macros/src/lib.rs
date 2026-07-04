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
//!   parenthesized expression, or `conj(expr)` marking an adjoint operand.
//!   Each `expr` must evaluate to a `Tensor` or `&Tensor`.
//! - A label appearing on two operands is contracted; a label appearing
//!   twice on ONE operand is a partial trace of that operand (TensorKit
//!   `@tensor a[i, i; j]`); a label appearing once must be listed in the
//!   output. Violations are compile errors.
//! - Lowers to `tenet_network::contract_network` (planner IR directly; no
//!   einsum strings).

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

/// One RHS operand: `expr[labels]`, `(expr)[labels]` or `conj(expr)[labels]`.
struct Operand {
    tensor: Expr,
    conj: bool,
    group: LabelGroup,
}

impl Parse for Operand {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let (tensor, conj) = if input.peek(Ident) && input.peek2(syn::token::Paren) {
            let ident: Ident = input.parse()?;
            if ident != "conj" {
                return Err(syn::Error::new(
                    ident.span(),
                    "expected `conj(...)`, an identifier, or a parenthesized expression",
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
            let ident: Ident = input.parse()?;
            (syn::parse_quote!(#ident), false)
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

    let operands = parsed.operands.iter().map(|op| {
        let tensor = &op.tensor;
        let conj = op.conj;
        let labels = &op.group.labels;
        let split = option_tokens(op.group.split);
        quote! {
            ::tenet_network::NetOperand {
                tensor: &#tensor,
                conj: #conj,
                labels: &[#(#labels),*],
                codomain_split: #split,
            }
        }
    });
    let output = &parsed.output.labels;
    let out_split = option_tokens(parsed.output.split);

    quote! {
        ::tenet_network::contract_network(
            &[#(#operands),*],
            &[#(#output),*],
            #out_split,
        )
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
    use super::check_labels;

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
