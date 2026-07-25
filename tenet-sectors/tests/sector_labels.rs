use std::collections::BTreeSet;

use tenet_sectors::{product_sector, ProductSector, U1Irrep, Z2Irrep};

#[test]
fn product_sector_orders_lexicographically_by_component() {
    // What: the label order is the left component first, the right component
    // only as a tie-break, so a `Sector: Ord` bound sees a total order that
    // agrees with the pair order of its components.
    let low_low = product_sector(U1Irrep::new(-1), Z2Irrep::EVEN);
    let low_high = product_sector(U1Irrep::new(-1), Z2Irrep::ODD);
    let high_low = product_sector(U1Irrep::new(2), Z2Irrep::EVEN);

    assert!(low_low < low_high);
    assert!(low_high < high_low);
    assert_eq!(low_low.cmp(&low_low), core::cmp::Ordering::Equal);
}

#[test]
fn product_sector_is_usable_as_a_btree_key() {
    // What: `Ord` is real enough for the ordered containers the typed facade
    // needs, including deduplication of equal labels.
    let labels: BTreeSet<ProductSector<U1Irrep, Z2Irrep>> = [
        product_sector(U1Irrep::new(1), Z2Irrep::ODD),
        product_sector(U1Irrep::new(0), Z2Irrep::EVEN),
        product_sector(U1Irrep::new(1), Z2Irrep::EVEN),
        product_sector(U1Irrep::new(0), Z2Irrep::EVEN),
    ]
    .into_iter()
    .collect();

    assert_eq!(
        labels.into_iter().collect::<Vec<_>>(),
        vec![
            product_sector(U1Irrep::new(0), Z2Irrep::EVEN),
            product_sector(U1Irrep::new(1), Z2Irrep::EVEN),
            product_sector(U1Irrep::new(1), Z2Irrep::ODD),
        ]
    );
}
