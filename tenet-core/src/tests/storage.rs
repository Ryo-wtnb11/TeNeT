use super::*;

    #[test]
    fn host_execution_rejects_reported_extent_changes_before_callbacks_or_writes() {
        // What: safe interior mutability in external storage cannot make a
        // short or oversized reported extent reach callbacks, views, or writes.
        for reported_len in [1, 3] {
            let tensor = adversarial_host_tensor(2);
            tensor.storage().reported_len.set(reported_len);
            assert_host_execution_rejects_extent(
                tensor,
                CoreError::DimensionMismatch {
                    expected: 2,
                    actual: reported_len,
                },
            );
        }
    }

    #[test]
    fn vec_storage_allocates_similar_host_scratch() {
        let storage = vec![1.0_f64, 2.0];
        let scratch = storage.similar_filled(4, 0.5);

        assert_eq!(scratch, vec![0.5; 4]);
        assert_eq!(scratch.placement(), Placement::Host);
    }
