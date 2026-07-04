use tenet_core::{
    BlockKey, FusionTensorMapSpace, FusionTreeHomSpace, SU2FusionRule, TensorMap, TensorMapSpace,
};
use tenet_tensors::{
    tensorcontract_fusion_explicit_plan, tensorcontract_fusion_explicit_plan_into,
    tensorcontract_fusion_explicit_plan_into_canonical_dst, tree_transform_into_with_context,
    OutputAxisOrder, TensorContractSpec, TreeTransformBuiltinRuleCacheKey,
    TreeTransformExecutionContext,
};

fn main() {
    let rule = SU2FusionRule;
    let lhs_hom = FusionTreeHomSpace::from_sector_ids([1, 1, 1], [1]);
    let rhs_hom = FusionTreeHomSpace::from_sector_ids([1], [1, 1, 1]);
    let axes = TensorContractSpec::with_default_output_order(&[0, 1, 2], &[1, 2, 3]);
    let tensorkit_axes =
        TensorContractSpec::new(&[0, 1, 2], &[1, 2, 3], OutputAxisOrder::from_axes(&[1, 0]));
    let lhs_canonical_hom = lhs_hom
        .permute(&rule, &[3], &[0, 1, 2])
        .expect("valid lhs canonical tree-pair transform");
    let rhs_canonical_hom = rhs_hom
        .permute(&rule, &[1, 2, 3], &[0])
        .expect("valid rhs canonical tree-pair transform");
    let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        &lhs_hom,
        &rhs_hom,
        axes.lhs_contracting_axes(),
        axes.rhs_contracting_axes(),
        &[0, 1],
        1,
    )
    .unwrap();

    println!("lhs_keys");
    for (i, key) in lhs_hom.fusion_tree_keys(&rule).iter().enumerate() {
        println!("{i}: {key:?}");
    }
    println!("rhs_keys");
    for (i, key) in rhs_hom.fusion_tree_keys(&rule).iter().enumerate() {
        println!("{i}: {key:?}");
    }
    println!("lhs_canonical_keys");
    for (i, key) in lhs_canonical_hom.fusion_tree_keys(&rule).iter().enumerate() {
        println!("{i}: {key:?}");
    }
    println!("rhs_canonical_keys");
    for (i, key) in rhs_canonical_hom.fusion_tree_keys(&rule).iter().enumerate() {
        println!("{i}: {key:?}");
    }

    let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<3, 1>::from_dims([2, 2, 2], [2]).unwrap(),
        lhs_hom,
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 3>::from_dims([2], [2, 2, 2]).unwrap(),
        rhs_hom,
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let lhs_canonical_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 3>::from_dims([2], [2, 2, 2]).unwrap(),
        lhs_canonical_hom,
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let rhs_canonical_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<3, 1>::from_dims([2, 2, 2], [2]).unwrap(),
        rhs_canonical_hom,
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap(),
        dst_hom,
        &rule,
        [vec![2, 2]],
    )
    .unwrap();
    print_structure("lhs_structure", lhs_space.subblock_structure());
    print_structure("rhs_structure", rhs_space.subblock_structure());
    print_structure(
        "lhs_canonical_structure",
        lhs_canonical_space.subblock_structure(),
    );
    print_structure(
        "rhs_canonical_structure",
        rhs_canonical_space.subblock_structure(),
    );
    print_structure("dst_structure", dst_space.subblock_structure());

    let lhs_data = (0..32)
        .map(|index| 1.0 + 0.125 * index as f64)
        .collect::<Vec<_>>();
    let rhs_data = (0..32)
        .map(|index| -3.0 + 0.25 * index as f64)
        .collect::<Vec<_>>();
    let lhs = TensorMap::<f64, 3, 1>::from_vec_with_fusion_space(lhs_data, lhs_space).unwrap();
    let rhs = TensorMap::<f64, 1, 3>::from_vec_with_fusion_space(rhs_data, rhs_space).unwrap();
    let mut lhs_canonical = TensorMap::<f64, 1, 3>::from_vec_with_fusion_space(
        vec![0.0; lhs_canonical_space.required_len().unwrap()],
        lhs_canonical_space.clone(),
    )
    .unwrap();
    let mut rhs_canonical = TensorMap::<f64, 3, 1>::from_vec_with_fusion_space(
        vec![0.0; rhs_canonical_space.required_len().unwrap()],
        rhs_canonical_space.clone(),
    )
    .unwrap();
    let mut dst =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![2.0, -1.0, 4.0, -3.0], dst_space)
            .unwrap();
    let plan = tensorcontract_fusion_explicit_plan(
        &rule,
        dst.fusion_space().unwrap(),
        lhs.fusion_space().unwrap(),
        rhs.fusion_space().unwrap(),
        axes,
    )
    .unwrap();
    let mut tree_context =
        TreeTransformExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
    tree_transform_into_with_context(
        &mut tree_context,
        &rule,
        plan.lhs_transform().clone(),
        &mut lhs_canonical,
        &lhs,
        1.0,
        0.0,
    )
    .unwrap();
    tree_transform_into_with_context(
        &mut tree_context,
        &rule,
        plan.rhs_transform().clone(),
        &mut rhs_canonical,
        &rhs,
        1.0,
        0.0,
    )
    .unwrap();
    println!("lhs_canonical_data {:?}", lhs_canonical.data());
    println!("rhs_canonical_data {:?}", rhs_canonical.data());
    tensorcontract_fusion_explicit_plan_into(
        &rule,
        &plan,
        &mut dst,
        &mut lhs_canonical,
        &mut rhs_canonical,
        &lhs,
        &rhs,
        -1.5,
        0.25,
    )
    .unwrap();
    println!("dst_data {:?}", dst.data());
    println!(
        "dst_checksum {:.12}",
        dst.data().iter().copied().sum::<f64>()
    );

    let tensorkit_dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        lhs.fusion_space().unwrap().homspace(),
        rhs.fusion_space().unwrap().homspace(),
        tensorkit_axes.lhs_contracting_axes(),
        tensorkit_axes.rhs_contracting_axes(),
        &[1, 0],
        1,
    )
    .unwrap();
    let tensorkit_dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap(),
        tensorkit_dst_hom,
        &rule,
        [vec![2, 2]],
    )
    .unwrap();
    print_structure(
        "tensorkit_order_dst_structure",
        tensorkit_dst_space.subblock_structure(),
    );
    let mut tensorkit_order_dst = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
        vec![2.0, -1.0, 4.0, -3.0],
        tensorkit_dst_space,
    )
    .unwrap();
    let tensorkit_order_plan = tensorcontract_fusion_explicit_plan(
        &rule,
        tensorkit_order_dst.fusion_space().unwrap(),
        lhs.fusion_space().unwrap(),
        rhs.fusion_space().unwrap(),
        tensorkit_axes,
    )
    .unwrap();
    let mut lhs_canonical = TensorMap::<f64, 1, 3>::from_vec_with_fusion_space(
        vec![0.0; lhs_canonical_space.required_len().unwrap()],
        lhs_canonical_space,
    )
    .unwrap();
    let mut rhs_canonical = TensorMap::<f64, 3, 1>::from_vec_with_fusion_space(
        vec![0.0; rhs_canonical_space.required_len().unwrap()],
        rhs_canonical_space,
    )
    .unwrap();
    let canonical_dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        lhs.fusion_space().unwrap().homspace(),
        rhs.fusion_space().unwrap().homspace(),
        axes.lhs_contracting_axes(),
        axes.rhs_contracting_axes(),
        &[0, 1],
        1,
    )
    .unwrap();
    let canonical_dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap(),
        canonical_dst_hom,
        &rule,
        [vec![2, 2]],
    )
    .unwrap();
    let mut canonical_dst = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
        vec![0.0; canonical_dst_space.required_len().unwrap()],
        canonical_dst_space,
    )
    .unwrap();
    tensorcontract_fusion_explicit_plan_into_canonical_dst(
        &rule,
        &tensorkit_order_plan,
        &mut tensorkit_order_dst,
        &mut canonical_dst,
        &mut lhs_canonical,
        &mut rhs_canonical,
        &lhs,
        &rhs,
        -1.5,
        0.25,
    )
    .unwrap();
    println!("tensorkit_order_dst_data {:?}", tensorkit_order_dst.data());
    println!(
        "tensorkit_order_dst_checksum {:.12}",
        tensorkit_order_dst.data().iter().copied().sum::<f64>()
    );
}

fn print_structure(label: &str, structure: &tenet_core::BlockStructure) {
    println!("{label}");
    for i in 0..structure.block_count() {
        let block = structure.block(i).unwrap();
        match block.key() {
            BlockKey::FusionTree(key) => {
                println!(
                    "{i}: offset={} shape={:?} key={key:?}",
                    block.offset(),
                    block.shape()
                );
            }
            key => {
                println!(
                    "{i}: offset={} shape={:?} key={key:?}",
                    block.offset(),
                    block.shape()
                );
            }
        }
    }
}
