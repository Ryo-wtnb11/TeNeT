//! Labeled networks for tests and examples, written with the explicit
//! `tenet_network::Network` API (#2022).
//!
//! Include with `#[path = ".../tests/support/network.rs"] mod network;`.
//! An operand is its codomain and domain labels, as `t[i, j; k]` writes them:
//! `op(&["i", "j"], &["k"])`; `conj(op(..))` reads it through its adjoint.
//! `net(&[op(..), ..], &[codomain..], &[domain..])` builds the network,
//! and `.contract(&[&t, ..])` runs it through the Runtime's plan cache.
//! `flat(&[..])` writes an operand's labels without asserting its `;` split.

#![allow(dead_code)]

use tenet_network::{Network, TemporaryLabel};

/// One operand's written labels, codomain then domain.
#[derive(Clone, Copy)]
pub struct Operand<'a> {
    codomain: &'a [&'a str],
    domain: &'a [&'a str],
    split: bool,
    conj: bool,
}

/// The operand `t[codomain; domain]`.
pub fn op<'a>(codomain: &'a [&'a str], domain: &'a [&'a str]) -> Operand<'a> {
    Operand {
        codomain,
        domain,
        split: true,
        conj: false,
    }
}

/// The operand `t[labels]`, its labels in flat leg order with no `;`: the
/// tensor's codomain rank is not checked against the labels.
pub fn flat<'a>(labels: &'a [&'a str]) -> Operand<'a> {
    Operand {
        codomain: labels,
        domain: &[],
        split: false,
        conj: false,
    }
}

/// The operand read through its adjoint, `conj(t)[codomain; domain]`, with
/// the labels written for the stored tensor's legs.
pub fn conj(operand: Operand<'_>) -> Operand<'_> {
    Operand {
        conj: true,
        ..operand
    }
}

/// The network `[output_codomain; output_domain] = operands[0] * ...`.
///
/// # Panics
///
/// When the labels are not a valid network (a label on one operand only and
/// not in the output, a label written twice on one operand, ...).
pub fn net(operands: &[Operand<'_>], output_codomain: &[&str], output_domain: &[&str]) -> Network {
    let labels = |names: &[&str]| -> Vec<TemporaryLabel> {
        names.iter().copied().map(TemporaryLabel::from).collect()
    };
    Network::new(
        operands
            .iter()
            .map(|operand| {
                let mut written = labels(operand.codomain);
                written.extend(labels(operand.domain));
                written
            })
            .collect(),
        operands.iter().map(|operand| operand.conj).collect(),
        operands
            .iter()
            .map(|operand| operand.split.then_some(operand.codomain.len()))
            .collect(),
        labels(output_codomain)
            .into_iter()
            .chain(labels(output_domain))
            .collect(),
        Some(output_codomain.len()),
    )
    .expect("valid test network")
}
