//! Optimizer::AutoHq must survive networks the upstream `opt-einsum-path`
//! drivers reject (all-dim-1 gram topology from finite-torus CTRG) by
//! falling back down the legacy auto-hq -> auto -> dp -> greedy chain.
#![cfg(feature = "opt-path")]

use tenet::prelude::*;
use tenet_network::tensor;

#[test]
fn autohq_falls_back_on_all_dim1_gram_topology() {
    // Upstream repro: dp/auto-hq error with "No contraction found for given
    // memory_limit" on this 6-operand topology when every dimension is 1.
    let r = opt_einsum_path::contract_path(
        "abc,dce,bfgh,ghdi,jkf,kil->ajel",
        &[
            vec![1usize, 1, 1],
            vec![1, 1, 1],
            vec![1, 1, 1, 1],
            vec![1, 1, 1, 1],
            vec![1, 1, 1],
            vec![1, 1, 1],
        ],
        "auto-hq",
        opt_einsum_path::typing::SizeLimitType::None,
    );
    assert!(
        r.is_err(),
        "upstream fixed the dim-1 bug; drop the fallback?"
    );

    // The user-layer AutoHq optimizer must still contract such a network.
    let rt = Runtime::builder()
        .plan_cache(PlanCacheConfig {
            optimizer: Optimizer::AutoHq,
            ..PlanCacheConfig::default()
        })
        .build()
        .unwrap();
    let v = Space::su2([(0, 1)]);
    let cne = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v], 1).unwrap();
    let sne = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v], 2).unwrap();
    let ev = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v, &v], 3).unwrap();
    let out = tensor!(
        [o1, o2; o3, o4] = cne[n3, n4; o3]
            * conj(cne)[n3, n5; o1]
            * ev[n1, n2; n7, n4]
            * conj(ev)[n1, n2; n8, n5]
            * sne[n7, n6; o4]
            * conj(sne)[n8, n6; o2]
    );
    assert!(out.is_ok(), "{:?}", out.err());
}
