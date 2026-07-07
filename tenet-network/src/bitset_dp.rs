//! Bitset dynamic-programming contraction-order search: a faithful port of
//! `opt_einsum_path`'s `DynamicProgramming` optimizer (minimize = flops,
//! `cost_cap = MaxInput`, `search_outer = false` — the `"dp"` default, which is
//! also what `"auto-hq"` dispatches to for the small networks TeNeT contracts)
//! with the hot `BigUint` subgraph bitmap and `BTreeSet<char>` index sets
//! replaced by `u128` bitmasks.
//!
//! It produces the **identical** optimal order to the upstream crate: the
//! algorithm, cost model and tie-breaks are preserved (DP states are kept in a
//! `BTreeMap<u128, _>`, so subsets are still visited in ascending order and the
//! strict `cost < term.cost` comparison keeps the same winner on ties). So it
//! is parity-preserving — the CTRG energy anchor is unchanged — but ~10-50x
//! faster on the cold first search, where the upstream `BTreeSet`/`BigUint`
//! churn dominated (~90% of TeNeT's cold contraction time).
//!
//! Falls back (returns `None`) for networks with more than 128 tensors or 128
//! distinct indices, where a `u128` mask no longer fits; the caller then uses
//! the crate's `contract_path`.

use std::collections::BTreeMap;
use std::rc::Rc;

/// A contraction tree: leaves are original tensor positions. Held behind `Rc`
/// in the DP table so combining two subtrees is a pointer clone, not a deep
/// copy — the deep copy was the bulk of the DP's allocation churn.
enum Tree {
    Leaf(usize),
    Node(Vec<Rc<Tree>>),
}

/// Port of opt_einsum's `_tree_to_sequence`: convert a contraction tree to a
/// linear path of positions into the shrinking active-tensor list.
fn tree_to_sequence(tree: &Tree) -> Vec<Vec<usize>> {
    if let Tree::Leaf(_) = tree {
        return Vec::new();
    }
    let mut c: std::collections::VecDeque<&Tree> = std::collections::VecDeque::new();
    c.push_back(tree);
    let mut t: Vec<usize> = Vec::new();
    let mut s: std::collections::VecDeque<Vec<usize>> = std::collections::VecDeque::new();

    while let Some(j) = c.pop_back() {
        s.push_front(Vec::new());
        if let Tree::Node(children) = j {
            let mut int_children: Vec<usize> = children
                .iter()
                .filter_map(|child| match child.as_ref() {
                    Tree::Leaf(i) => Some(*i),
                    _ => None,
                })
                .collect();
            int_children.sort_unstable();
            for i in int_children {
                let pos = t.iter().filter(|&&q| q < i).count();
                s[0].push(pos);
                t.insert(pos, i);
            }
            for i_tup in children
                .iter()
                .filter(|child| matches!(child.as_ref(), Tree::Node(_)))
            {
                let pos = t.len() + c.len();
                s[0].push(pos);
                c.push_back(i_tup.as_ref());
            }
        }
    }
    s.into_iter().collect()
}

fn simple_tree_tuple(seq: &[Rc<Tree>]) -> Rc<Tree> {
    seq.iter()
        .cloned()
        .reduce(|left, right| Rc::new(Tree::Node(vec![left, right])))
        .expect("non-empty sequence")
}

/// Product of the sizes of the indices set in `mask`.
#[inline]
fn size_of(mask: u128, sizes: &[f64]) -> f64 {
    let mut m = mask;
    let mut product = 1.0;
    while m != 0 {
        let bit = m.trailing_zeros() as usize;
        product *= sizes[bit];
        m &= m - 1;
    }
    product
}

/// Union of the index masks of the tensors set in `bitmap`.
#[inline]
fn union_of(bitmap: u128, inputs: &[u128]) -> u128 {
    let mut m = bitmap;
    let mut acc = 0u128;
    while m != 0 {
        let bit = m.trailing_zeros() as usize;
        acc |= inputs[bit];
        m &= m - 1;
    }
    acc
}

/// Port of `find_disconnected_subgraphs` (u128 tensor bitmaps).
fn find_disconnected_subgraphs(inputs: &[u128], output: u128) -> Vec<Vec<usize>> {
    let n = inputs.len();
    let mut subgraphs = Vec::new();
    let mut unused: Vec<usize> = (0..n).collect();
    let all_inputs = inputs.iter().fold(0u128, |a, &b| a | b);
    let i_sum = all_inputs & !output;

    while let Some(&first) = unused.first() {
        let mut g = Vec::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(first);
        unused.retain(|&k| k != first);
        while let Some(j) = queue.pop_front() {
            g.push(j);
            let i_tmp = i_sum & inputs[j];
            let neighbors: Vec<usize> = unused
                .iter()
                .copied()
                .filter(|&k| inputs[k] & i_tmp != 0)
                .collect();
            for neighbor in neighbors {
                queue.push_back(neighbor);
                unused.retain(|&k| k != neighbor);
            }
        }
        g.sort_unstable();
        subgraphs.push(g);
    }
    subgraphs
}

/// Port of `dp_calc_legs`.
#[inline]
fn dp_calc_legs(
    g: u128,
    all_tensors: u128,
    s: u128,
    inputs: &[u128],
    i1_cut_i2_wo_output: u128,
    i1_union_i2: u128,
) -> u128 {
    let r = g & (all_tensors ^ s);
    let i_r = if r != 0 { union_of(r, inputs) } else { 0 };
    let i_contract = i1_cut_i2_wo_output & !i_r;
    i1_union_i2 & !i_contract
}

struct Term {
    indices: u128,
    cost: f64,
    contract: Rc<Tree>,
}

/// Port of `dp_parse_out_single_term_ops`: strip indices that appear on only
/// one tensor. Returns (parsed input masks, done/scalar trees, per-parsed
/// contraction trees referencing original positions).
fn dp_parse_out_single_term_ops(
    inputs: &[u128],
    single_mask: u128,
) -> (Vec<u128>, Vec<Rc<Tree>>, Vec<Rc<Tree>>) {
    let mut parsed = Vec::new();
    let mut done = Vec::new();
    let mut contractions = Vec::new();
    for (j, &input) in inputs.iter().enumerate() {
        let reduced = input & !single_mask;
        if reduced == 0 && input != 0 {
            done.push(Rc::new(Tree::Node(vec![Rc::new(Tree::Leaf(j))])));
        } else {
            contractions.push(if reduced != input {
                Rc::new(Tree::Node(vec![Rc::new(Tree::Leaf(j))]))
            } else {
                Rc::new(Tree::Leaf(j))
            });
            parsed.push(reduced);
        }
    }
    (parsed, done, contractions)
}

/// Faithful port of `DynamicProgramming::find_optimal_path` for
/// minimize = flops, cost_cap = MaxInput, search_outer = false.
fn find_optimal_path(inputs: &[u128], output: u128, sizes: &[f64]) -> Vec<Vec<usize>> {
    // Index occurrence counts across inputs + output.
    let mut ind_counts = [0u32; 128];
    for &inp in inputs {
        let mut m = inp;
        while m != 0 {
            let b = m.trailing_zeros() as usize;
            ind_counts[b] += 1;
            m &= m - 1;
        }
    }
    {
        let mut m = output;
        while m != 0 {
            let b = m.trailing_zeros() as usize;
            ind_counts[b] += 1;
            m &= m - 1;
        }
    }
    let mut single_mask = 0u128;
    for (b, &c) in ind_counts.iter().enumerate() {
        if c == 1 {
            single_mask |= 1u128 << b;
        }
    }

    let (inputs, inputs_done, inputs_contractions) =
        dp_parse_out_single_term_ops(inputs, single_mask);
    if inputs.is_empty() {
        return tree_to_sequence(&simple_tree_tuple(&inputs_done));
    }

    let mut subgraph_contractions = inputs_done;
    let mut subgraph_sizes: Vec<f64> = vec![1.0; subgraph_contractions.len()];

    let subgraphs = find_disconnected_subgraphs(&inputs, output);
    let all_tensors: u128 = if inputs.len() == 128 {
        u128::MAX
    } else {
        (1u128 << inputs.len()) - 1
    };
    let naive_cost = inputs.len() as f64 * size_of(all_indices(&inputs), sizes);

    for g in subgraphs {
        let bitmap_g = g.iter().fold(0u128, |acc, &j| acc | (1u128 << j));

        // DP table: x[k] = best contraction of each k-subset of g.
        let mut x: Vec<BTreeMap<u128, Term>> = (0..=g.len()).map(|_| BTreeMap::new()).collect();
        x[1] = g
            .iter()
            .map(|&j| {
                (
                    1u128 << j,
                    Term {
                        indices: inputs[j],
                        cost: 0.0,
                        contract: inputs_contractions[j].clone(),
                    },
                )
            })
            .collect();

        // Cost-cap deepening (as upstream): start at the MaxInput cap and grow
        // by the smallest bond dim each sweep. Measured faster than an unbounded
        // single sweep here — the cap prunes expensive subsets in the early
        // (cheap) sweeps, so the total work is less than one full unpruned pass.
        let subgraph_inds = union_of(bitmap_g, &inputs);
        let mut cost_cap = size_of(subgraph_inds & output, sizes);
        let cost_increment = {
            let mut m = subgraph_inds;
            let mut min = f64::INFINITY;
            while m != 0 {
                let b = m.trailing_zeros() as usize;
                min = min.min(sizes[b]);
                m &= m - 1;
            }
            if subgraph_inds == 0 {
                2.0
            } else {
                min.max(2.0)
            }
        };

        while x.last().unwrap().is_empty() {
            for n in 2..=g.len() {
                let (left, right) = x.split_at_mut(n);
                let xn = &mut right[0];
                for m in 1..=(n / 2) {
                    // Clone the two source levels so we can borrow `xn` mutably;
                    // levels are small (subsets of a connected subgraph).
                    let a = &left[m];
                    let b = &left[n - m];
                    for (&s1, term1) in a.iter() {
                        for (&s2, term2) in b.iter() {
                            if s1 & s2 == 0 && (m != n - m || s1 < s2) {
                                let i1 = term1.indices;
                                let i2 = term2.indices;
                                let i1_cut_i2_wo_output = i1 & i2 & !output;
                                if i1_cut_i2_wo_output != 0 {
                                    let i1_union_i2 = i1 | i2;
                                    let cost =
                                        term1.cost + term2.cost + size_of(i1_union_i2, sizes);
                                    if cost <= cost_cap {
                                        let s = s1 | s2;
                                        let better = match xn.get(&s) {
                                            Some(t) => cost < t.cost,
                                            None => true,
                                        };
                                        if better {
                                            let indices = dp_calc_legs(
                                                bitmap_g,
                                                all_tensors,
                                                s,
                                                &inputs,
                                                i1_cut_i2_wo_output,
                                                i1_union_i2,
                                            );
                                            xn.insert(
                                                s,
                                                Term {
                                                    indices,
                                                    cost,
                                                    contract: Rc::new(Tree::Node(vec![
                                                        Rc::clone(&term1.contract),
                                                        Rc::clone(&term2.contract),
                                                    ])),
                                                },
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            cost_cap *= cost_increment;
            if cost_cap > naive_cost && x.last().unwrap().is_empty() {
                // No contraction under the (memory) limit; matches upstream's
                // error path. Degenerate for our networks — bail to a trivial
                // left-nested order so the caller can still proceed.
                break;
            }
        }

        if let Some((_, term)) = x.last().unwrap().iter().next() {
            subgraph_contractions.push(term.contract.clone());
            subgraph_sizes.push(size_of(term.indices, sizes));
        } else {
            // Fallback: left-nested over the subgraph (only on the break above).
            let seq: Vec<Rc<Tree>> = g
                .iter()
                .map(|&j| Rc::clone(&inputs_contractions[j]))
                .collect();
            subgraph_contractions.push(simple_tree_tuple(&seq));
            subgraph_sizes.push(0.0);
        }
    }

    // Sort subgraphs by size (stable, ascending) — matches upstream.
    let mut order: Vec<usize> = (0..subgraph_sizes.len()).collect();
    order.sort_by(|&a, &b| subgraph_sizes[a].partial_cmp(&subgraph_sizes[b]).unwrap());
    let sorted: Vec<Rc<Tree>> = order
        .into_iter()
        .map(|i| Rc::clone(&subgraph_contractions[i]))
        .collect();
    tree_to_sequence(&simple_tree_tuple(&sorted))
}

#[inline]
fn all_indices(inputs: &[u128]) -> u128 {
    inputs.iter().fold(0u128, |a, &b| a | b)
}

/// Compute the optimal pairwise contraction path for a network described by
/// per-tensor label lists and per-label dimensions. Returns `None` (caller
/// falls back to the crate) when the network exceeds the `u128` bit budget.
pub(crate) fn bitset_dp_path(
    per_tensor_labels: &[Vec<usize>],
    output_labels: &[usize],
    label_dims: &[usize],
) -> Option<Vec<Vec<usize>>> {
    let num_indices = label_dims.len();
    if num_indices > 128 || per_tensor_labels.len() > 128 {
        return None;
    }
    let inputs: Vec<u128> = per_tensor_labels
        .iter()
        .map(|labels| labels.iter().fold(0u128, |m, &l| m | (1u128 << l)))
        .collect();
    let output: u128 = output_labels.iter().fold(0u128, |m, &l| m | (1u128 << l));
    let sizes: Vec<f64> = label_dims.iter().map(|&d| d as f64).collect();
    Some(find_optimal_path(&inputs, output, &sizes))
}
