use super::*;

#[test]
fn vec_storage_allocates_similar_host_scratch() {
    let storage = vec![1.0_f64, 2.0];
    let scratch = storage.similar_filled(4, 0.5);

    assert_eq!(scratch, vec![0.5; 4]);
    assert_eq!(scratch.placement(), Placement::Host);
}
