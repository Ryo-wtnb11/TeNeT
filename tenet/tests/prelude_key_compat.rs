#[allow(deprecated)]
#[test]
fn prelude_retains_the_legacy_fusion_tree_block_key_name() {
    // What: existing prelude imports continue to resolve while new code uses
    // the categorical FusionTreePairKey name.
    use tenet::prelude::{FusionTreeBlockKey, FusionTreePairKey};

    fn accepts_pair(_: Option<FusionTreePairKey>) {}
    let legacy: Option<FusionTreeBlockKey> = None;
    accepts_pair(legacy);
}
