use std::convert::Infallible;
use std::sync::{mpsc, Arc, Barrier};
use std::time::Duration;

use num_complex::Complex64;
use tenet_core::{BlockKey, BlockSpec, BlockStructure, FusionTreeHomSpace, RuleIdentity};

use super::{
    RuntimeTreeTransformCacheLedger, RuntimeTreeTransformKey, RuntimeTreeTransformOperationKey,
    RuntimeTreeTransformStore,
};
use crate::{
    TreeTransformBlockSpec, TreeTransformOperation, TreeTransformStructure,
    TreeTransformStructureCacheKey,
};

struct TestRuleIdentity;

fn fixture(
    tag: usize,
) -> (
    RuntimeTreeTransformKey,
    Arc<TreeTransformStructure<f64>>,
    BlockStructure,
) {
    fixture_with_rule(tag, RuleIdentity::of_type::<TestRuleIdentity>())
}

fn fixture_with_rule(
    tag: usize,
    rule: RuleIdentity,
) -> (
    RuntimeTreeTransformKey,
    Arc<TreeTransformStructure<f64>>,
    BlockStructure,
) {
    let block = BlockSpec::with_key(BlockKey::ordinal(tag), vec![1], vec![1], 0).unwrap();
    let structure = BlockStructure::from_blocks_with_rank(1, vec![block]).unwrap();
    let compiled = Arc::new(
        TreeTransformStructure::compile_structures(
            &structure,
            &structure,
            &[TreeTransformBlockSpec::single(0, 0, 1.0)],
        )
        .unwrap(),
    );
    let key = TreeTransformStructureCacheKey::from_structures(
        RuntimeTreeTransformOperationKey {
            rule,
            operation: TreeTransformOperation::permute([tag], []),
            orientation: tenet_core::FusionTreePairOrientation::Direct,
            logical_source: None,
        },
        &structure,
        &structure,
    )
    .unwrap();
    (key, compiled, structure)
}

fn complex_fixture(
    tag: usize,
) -> (
    RuntimeTreeTransformKey,
    Arc<TreeTransformStructure<Complex64>>,
) {
    let block = BlockSpec::with_key(BlockKey::ordinal(tag), vec![1], vec![1], 0).unwrap();
    let structure = BlockStructure::from_blocks_with_rank(1, vec![block]).unwrap();
    let compiled = Arc::new(
        TreeTransformStructure::compile_structures(
            &structure,
            &structure,
            &[TreeTransformBlockSpec::single(
                0,
                0,
                Complex64::new(0.0, 1.0),
            )],
        )
        .unwrap(),
    );
    let key = TreeTransformStructureCacheKey::from_structures(
        RuntimeTreeTransformOperationKey {
            rule: RuleIdentity::of_type::<TestRuleIdentity>(),
            operation: TreeTransformOperation::permute([tag], []),
            orientation: tenet_core::FusionTreePairOrientation::Direct,
            logical_source: None,
        },
        &structure,
        &structure,
    )
    .unwrap();
    (key, compiled)
}

fn pair_fixture(
    dst_tag: usize,
    src_tag: usize,
) -> (
    RuntimeTreeTransformKey,
    Arc<TreeTransformStructure<f64>>,
    BlockStructure,
    BlockStructure,
) {
    let structure = |tag| {
        let block = BlockSpec::with_key(BlockKey::ordinal(tag), vec![1], vec![1], 0).unwrap();
        BlockStructure::from_blocks_with_rank(1, vec![block]).unwrap()
    };
    let dst = structure(dst_tag);
    let src = structure(src_tag);
    let compiled = Arc::new(
        TreeTransformStructure::compile_structures(
            &dst,
            &src,
            &[TreeTransformBlockSpec::single(0, 0, 1.0)],
        )
        .unwrap(),
    );
    let key = TreeTransformStructureCacheKey::from_structures(
        RuntimeTreeTransformOperationKey {
            rule: RuleIdentity::of_type::<TestRuleIdentity>(),
            operation: TreeTransformOperation::permute([0], []),
            orientation: tenet_core::FusionTreePairOrientation::Direct,
            logical_source: None,
        },
        &dst,
        &src,
    )
    .unwrap();
    (key, compiled, dst, src)
}

#[test]
fn shared_ledger_preserves_full_capacity_for_one_dtype() {
    // What: adding a second typed store does not partition the configured
    // resources; an f64-only workload can still occupy the complete cache.
    let (key0, structure0, _) = fixture(40);
    let (key1, structure1, _) = fixture(41);
    let charge0 = RuntimeTreeTransformStore::<f64>::charged_entry_bytes(&key0, &structure0);
    let charge1 = RuntimeTreeTransformStore::<f64>::charged_entry_bytes(&key1, &structure1);
    let ledger = Arc::new(RuntimeTreeTransformCacheLedger::with_limits(
        2,
        charge0.saturating_add(charge1),
    ));
    let real = RuntimeTreeTransformStore::with_runtime_ledger(Arc::clone(&ledger));
    let complex = RuntimeTreeTransformStore::<Complex64>::with_runtime_ledger(Arc::clone(&ledger));

    real.get_or_compile(key0, || Ok::<_, Infallible>(structure0))
        .unwrap();
    real.get_or_compile(key1, || Ok::<_, Infallible>(structure1))
        .unwrap();

    let info = ledger.store_pair_info(&real, &complex);
    assert_eq!(info.entries(), 2);
    assert_eq!(info.entry_capacity(), 2);
    assert_eq!(info.byte_budget(), charge0 + charge1);
    assert_eq!(info.admission_bypasses(), 0);
}

#[test]
fn pair_info_samples_one_store_once() {
    // What: accidentally passing one typed store twice completes without
    // deadlocking or double-counting its local metrics.
    let ledger = Arc::new(RuntimeTreeTransformCacheLedger::with_limits(2, usize::MAX));
    let store = RuntimeTreeTransformStore::with_runtime_ledger(Arc::clone(&ledger));
    let (key, structure, _) = fixture(45);
    store
        .get_or_compile(key, || Ok::<_, Infallible>(structure))
        .unwrap();

    let info = ledger.store_pair_info(&store, &store);
    assert_eq!(info.entries(), 1);
    assert_eq!(info.misses(), 1);
}

#[test]
fn reverse_pair_info_calls_complete_concurrently() {
    // What: opposite reporting order never holds both typed-store locks,
    // so it cannot form an ABBA cycle.
    let ledger = Arc::new(RuntimeTreeTransformCacheLedger::with_limits(2, usize::MAX));
    let real = Arc::new(RuntimeTreeTransformStore::<f64>::with_runtime_ledger(
        Arc::clone(&ledger),
    ));
    let complex = Arc::new(RuntimeTreeTransformStore::<Complex64>::with_runtime_ledger(
        Arc::clone(&ledger),
    ));
    let start = Arc::new(Barrier::new(3));
    let (done_tx, done_rx) = mpsc::channel();

    let first = {
        let ledger = Arc::clone(&ledger);
        let real = Arc::clone(&real);
        let complex = Arc::clone(&complex);
        let start = Arc::clone(&start);
        let done = done_tx.clone();
        std::thread::spawn(move || {
            start.wait();
            for _ in 0..5_000 {
                let _ = ledger.store_pair_info(real.as_ref(), complex.as_ref());
            }
            done.send(()).unwrap();
        })
    };
    let second = {
        let ledger = Arc::clone(&ledger);
        let real = Arc::clone(&real);
        let complex = Arc::clone(&complex);
        let start = Arc::clone(&start);
        let done = done_tx;
        std::thread::spawn(move || {
            start.wait();
            for _ in 0..5_000 {
                let _ = ledger.store_pair_info(complex.as_ref(), real.as_ref());
            }
            done.send(()).unwrap();
        })
    };
    start.wait();

    done_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    done_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    first.join().unwrap();
    second.join().unwrap();
}

#[test]
fn concurrent_cross_dtype_admission_obeys_one_total_byte_budget() {
    // What: the two typed stores race through one atomic cold-path ledger;
    // exactly one entry fits and the other admission bypasses.
    let (real_key, real_structure, _) = fixture(42);
    let (complex_key, complex_structure) = complex_fixture(42);
    let real_charge =
        RuntimeTreeTransformStore::<f64>::charged_entry_bytes(&real_key, &real_structure);
    let complex_charge = RuntimeTreeTransformStore::<Complex64>::charged_entry_bytes(
        &complex_key,
        &complex_structure,
    );
    let budget = real_charge.max(complex_charge);
    let ledger = Arc::new(RuntimeTreeTransformCacheLedger::with_limits(2, budget));
    let real = Arc::new(RuntimeTreeTransformStore::with_runtime_ledger(Arc::clone(
        &ledger,
    )));
    let complex = Arc::new(RuntimeTreeTransformStore::with_runtime_ledger(Arc::clone(
        &ledger,
    )));
    let start = Arc::new(Barrier::new(3));

    let real_worker = {
        let store = Arc::clone(&real);
        let start = Arc::clone(&start);
        std::thread::spawn(move || {
            start.wait();
            store
                .get_or_compile(real_key, || Ok::<_, Infallible>(real_structure))
                .unwrap();
        })
    };
    let complex_worker = {
        let store = Arc::clone(&complex);
        let start = Arc::clone(&start);
        std::thread::spawn(move || {
            start.wait();
            store
                .get_or_compile(complex_key, || Ok::<_, Infallible>(complex_structure))
                .unwrap();
        })
    };
    start.wait();
    real_worker.join().unwrap();
    complex_worker.join().unwrap();

    let info = ledger.store_pair_info(real.as_ref(), complex.as_ref());
    assert_eq!(info.entries(), 1);
    assert!(info.charged_payload_bytes() <= budget);
    assert_eq!(info.byte_budget(), budget);
    assert_eq!(info.misses(), 2);
    assert_eq!(info.admission_bypasses(), 1);
}

#[test]
fn coefficient_dtype_and_rule_identity_have_independent_keys() {
    // What: equal RuleIdentity values in distinct Rust store types cannot
    // cross-hit, while distinct identities in one dtype retain two entries.
    struct OtherRuleIdentity;

    let ledger = Arc::new(RuntimeTreeTransformCacheLedger::with_limits(4, usize::MAX));
    let real = RuntimeTreeTransformStore::with_runtime_ledger(Arc::clone(&ledger));
    let complex = RuntimeTreeTransformStore::with_runtime_ledger(Arc::clone(&ledger));
    let (real_key, real_structure, _) = fixture(43);
    let (complex_key, complex_structure) = complex_fixture(43);
    real.get_or_compile(real_key.clone(), || {
        Ok::<_, Infallible>(Arc::clone(&real_structure))
    })
    .unwrap();
    complex
        .get_or_compile(complex_key.clone(), || {
            Ok::<_, Infallible>(Arc::clone(&complex_structure))
        })
        .unwrap();
    assert_eq!(real.info().misses(), 1);
    assert_eq!(complex.info().misses(), 1);

    let (other_key, other_structure, _) =
        fixture_with_rule(43, RuleIdentity::of_type::<OtherRuleIdentity>());
    real.get_or_compile(other_key.clone(), || {
        Ok::<_, Infallible>(Arc::clone(&other_structure))
    })
    .unwrap();
    real.get_or_compile(real_key, || -> Result<_, Infallible> {
        unreachable!("real entry is warm")
    })
    .unwrap();
    real.get_or_compile(other_key, || -> Result<_, Infallible> {
        unreachable!("other rule entry is warm")
    })
    .unwrap();
    complex
        .get_or_compile(complex_key, || -> Result<_, Infallible> {
            unreachable!("complex entry is warm")
        })
        .unwrap();

    let info = ledger.store_pair_info(&real, &complex);
    assert_eq!(info.entries(), 3);
    assert_eq!(info.misses(), 3);
    assert_eq!(info.hits(), 3);
}

#[test]
fn mixed_full_cache_turns_over_within_one_dtype() {
    // What: a full shared ledger can replace the requesting dtype's LRU
    // without evicting the other dtype's resident entry.
    let ledger = Arc::new(RuntimeTreeTransformCacheLedger::with_limits(2, usize::MAX));
    let real = RuntimeTreeTransformStore::with_runtime_ledger(Arc::clone(&ledger));
    let complex = RuntimeTreeTransformStore::with_runtime_ledger(Arc::clone(&ledger));
    let (old_key, old_structure, _) = fixture(46);
    let (new_key, new_structure, _) = fixture(47);
    let (complex_key, complex_structure) = complex_fixture(46);

    real.get_or_compile(old_key.clone(), || {
        Ok::<_, Infallible>(Arc::clone(&old_structure))
    })
    .unwrap();
    complex
        .get_or_compile(complex_key.clone(), || {
            Ok::<_, Infallible>(Arc::clone(&complex_structure))
        })
        .unwrap();
    real.get_or_compile(new_key.clone(), || {
        Ok::<_, Infallible>(Arc::clone(&new_structure))
    })
    .unwrap();

    real.get_or_compile(new_key, || -> Result<_, Infallible> {
        unreachable!("new real entry is warm")
    })
    .unwrap();
    let mut old_recompiled = false;
    real.get_or_compile(old_key, || {
        old_recompiled = true;
        Ok::<_, Infallible>(old_structure)
    })
    .unwrap();
    assert!(old_recompiled);
    complex
        .get_or_compile(complex_key, || -> Result<_, Infallible> {
            unreachable!("complex entry survives real turnover")
        })
        .unwrap();

    let info = ledger.store_pair_info(&real, &complex);
    assert_eq!(info.entries(), 2);
    assert_eq!(info.evictions(), 2);
    assert_eq!(info.admission_bypasses(), 0);
}

#[test]
fn shared_clear_blocks_old_generation_and_releases_all_charges() {
    // What: clear of both typed stores releases the shared ledger and a
    // compilation begun before clear cannot republish afterward.
    let ledger = Arc::new(RuntimeTreeTransformCacheLedger::with_limits(2, usize::MAX));
    let real = Arc::new(RuntimeTreeTransformStore::with_runtime_ledger(Arc::clone(
        &ledger,
    )));
    let complex = Arc::new(RuntimeTreeTransformStore::with_runtime_ledger(Arc::clone(
        &ledger,
    )));
    let (real_key, real_structure, _) = fixture(44);
    real.get_or_compile(real_key, || Ok::<_, Infallible>(real_structure))
        .unwrap();

    let (complex_key, complex_structure) = complex_fixture(44);
    let started = Arc::new(Barrier::new(2));
    let resume = Arc::new(Barrier::new(2));
    let worker = {
        let store = Arc::clone(&complex);
        let started = Arc::clone(&started);
        let resume = Arc::clone(&resume);
        std::thread::spawn(move || {
            store
                .get_or_compile(complex_key, || {
                    started.wait();
                    resume.wait();
                    Ok::<_, Infallible>(complex_structure)
                })
                .unwrap()
        })
    };

    started.wait();
    real.clear();
    complex.clear();
    resume.wait();
    assert_eq!(worker.join().unwrap().block_count(), 1);

    let info = ledger.store_pair_info(real.as_ref(), complex.as_ref());
    assert_eq!(info.entries(), 0);
    assert_eq!(info.charged_payload_bytes(), 0);
    assert_eq!(info.hits(), 0);
    assert_eq!(info.misses(), 0);
}

#[test]
fn runtime_store_charges_and_releases_dependent_structures() {
    // What: one content owner is charged once, distinct source/destination
    // content affects admission, and eviction/clear release both owners.
    let (same_key, same_compiled, _, _) = pair_fixture(10, 10);
    let (distinct_key, distinct_compiled, _, _) = pair_fixture(11, 12);
    let same_charge =
        RuntimeTreeTransformStore::<f64>::charged_entry_bytes(&same_key, &same_compiled);
    let distinct_charge =
        RuntimeTreeTransformStore::<f64>::charged_entry_bytes(&distinct_key, &distinct_compiled);
    assert_eq!(
        distinct_charge,
        same_charge.saturating_add(distinct_key.src().charged_retained_bytes())
    );

    let budgeted =
        RuntimeTreeTransformStore::with_limits(2, distinct_charge.saturating_sub(1), usize::MAX);
    budgeted
        .get_or_compile(same_key, || Ok::<_, Infallible>(same_compiled))
        .unwrap();
    budgeted
        .get_or_compile(distinct_key, || Ok::<_, Infallible>(distinct_compiled))
        .unwrap();
    assert_eq!(budgeted.info().entries(), 1);
    assert_eq!(budgeted.info().admission_bypasses(), 1);

    let store = RuntimeTreeTransformStore::with_limits(1, usize::MAX, usize::MAX);
    let (first_key, first_compiled, first_dst, first_src) = pair_fixture(20, 21);
    let first_dst_content = first_dst.content_key();
    let first_src_content = first_src.content_key();
    let first_dst_weak = Arc::downgrade(&first_dst_content);
    let first_src_weak = Arc::downgrade(&first_src_content);
    drop(first_dst_content);
    drop(first_src_content);
    drop(
        store
            .get_or_compile(first_key, || Ok::<_, Infallible>(first_compiled))
            .unwrap(),
    );
    drop(first_dst);
    drop(first_src);
    assert!(first_dst_weak.upgrade().is_some());
    assert!(first_src_weak.upgrade().is_some());

    let (second_key, second_compiled, second_dst, second_src) = pair_fixture(22, 23);
    let second_dst_content = second_dst.content_key();
    let second_src_content = second_src.content_key();
    let second_dst_weak = Arc::downgrade(&second_dst_content);
    let second_src_weak = Arc::downgrade(&second_src_content);
    drop(second_dst_content);
    drop(second_src_content);
    drop(
        store
            .get_or_compile(second_key, || Ok::<_, Infallible>(second_compiled))
            .unwrap(),
    );
    drop(second_dst);
    drop(second_src);
    assert!(first_dst_weak.upgrade().is_none());
    assert!(first_src_weak.upgrade().is_none());
    assert!(second_dst_weak.upgrade().is_some());
    assert!(second_src_weak.upgrade().is_some());

    store.clear();
    assert!(second_dst_weak.upgrade().is_none());
    assert!(second_src_weak.upgrade().is_none());
}

#[test]
fn runtime_store_enforces_resources_and_clear_keeps_returned_arcs_valid() {
    // What: entry and byte pressure evict, oversized entries bypass, and
    // clear resets accounting without invalidating caller-owned payloads.
    let (key0, structure0, _) = fixture(0);
    let (key1, structure1, _) = fixture(1);
    let charge0 = RuntimeTreeTransformStore::<f64>::charged_entry_bytes(&key0, &structure0);
    let charge1 = RuntimeTreeTransformStore::<f64>::charged_entry_bytes(&key1, &structure1);

    let entry_limited = RuntimeTreeTransformStore::with_limits(1, usize::MAX, usize::MAX);
    let active = entry_limited
        .get_or_compile(key0.clone(), || {
            Ok::<_, Infallible>(Arc::clone(&structure0))
        })
        .unwrap();
    entry_limited
        .get_or_compile(key1.clone(), || {
            Ok::<_, Infallible>(Arc::clone(&structure1))
        })
        .unwrap();
    assert_eq!(entry_limited.info().entries(), 1);
    assert_eq!(entry_limited.info().evictions(), 1);
    assert_eq!(active.block_count(), 1);

    let byte_limited = RuntimeTreeTransformStore::with_limits(
        2,
        charge0.saturating_add(charge1).saturating_sub(1),
        usize::MAX,
    );
    byte_limited
        .get_or_compile(key0.clone(), || {
            Ok::<_, Infallible>(Arc::clone(&structure0))
        })
        .unwrap();
    byte_limited
        .get_or_compile(key1, || Ok::<_, Infallible>(Arc::clone(&structure1)))
        .unwrap();
    assert_eq!(byte_limited.info().entries(), 1);
    assert_eq!(byte_limited.info().evictions(), 1);

    let oversized =
        RuntimeTreeTransformStore::with_limits(2, usize::MAX, charge0.saturating_sub(1));
    oversized
        .get_or_compile(key0.clone(), || {
            Ok::<_, Infallible>(Arc::clone(&structure0))
        })
        .unwrap();
    assert_eq!(oversized.info().entries(), 0);
    assert_eq!(oversized.info().admission_bypasses(), 1);

    let disabled = RuntimeTreeTransformStore::new(0);
    disabled
        .get_or_compile(key0, || Ok::<_, Infallible>(Arc::clone(&structure0)))
        .unwrap();
    assert_eq!(disabled.info().entries(), 0);
    assert_eq!(disabled.info().byte_budget(), 0);
    assert_eq!(disabled.info().admission_bypasses(), 1);

    entry_limited.clear();
    let cleared = entry_limited.info();
    assert_eq!(cleared.entries(), 0);
    assert_eq!(cleared.entry_capacity(), 1);
    assert_eq!(cleared.charged_payload_bytes(), 0);
    assert_eq!(cleared.byte_budget(), usize::MAX);
    assert_eq!(cleared.hits(), 0);
    assert_eq!(cleared.misses(), 0);
    assert_eq!(cleared.evictions(), 0);
    assert_eq!(cleared.admission_bypasses(), 0);
    assert_eq!(active.block_count(), 1);
}

#[test]
fn exact_layout_admission_requires_explicit_publication_and_clear_removes_it() {
    // What: an ordinary completed structure is not an exact typed-layout
    // proof; successful publication enables borrowed-operation reuse, and
    // clearing the one Runtime store removes both together.
    let (key, structure, layout) = fixture(0);
    let store = RuntimeTreeTransformStore::default();
    store
        .get_or_compile(key.clone(), || Ok::<_, Infallible>(structure))
        .unwrap();
    let source_homspace = FusionTreeHomSpace::from_sector_ids([(0, 1)], []);
    let destination_homspace = FusionTreeHomSpace::from_sector_ids([], [(0, 1)]);
    let source_id = source_homspace.id();
    let destination_id = destination_homspace.id();
    let rule = RuleIdentity::of_type::<TestRuleIdentity>();
    let operation = TreeTransformOperation::permute([0], []);
    let source_layout = [layout.content_id(), 1, 0];
    let destination_layout = [layout.content_id(), 0, 1];

    assert!(store
        .admitted_tree_pair_operation(
            &rule,
            &source_id,
            source_layout,
            &destination_id,
            destination_layout,
            |_| true,
        )
        .is_none());
    assert!(store
        .admit_exact_tree_pair_layout(
            rule.clone(),
            &operation,
            &layout,
            &layout,
            (&source_id, source_layout),
            (&destination_id, destination_layout),
        )
        .unwrap());
    assert_eq!(
        store
            .admitted_tree_pair_operation(
                &rule,
                &source_id,
                source_layout,
                &destination_id,
                destination_layout,
                |candidate| candidate == &operation,
            )
            .unwrap(),
        operation
    );
    let foreign_destination = FusionTreeHomSpace::from_sector_ids([], [(1, 1)]).id();
    assert!(store
        .admitted_tree_pair_operation(
            &rule,
            &source_id,
            source_layout,
            &foreign_destination,
            destination_layout,
            |_| true,
        )
        .is_none());

    store.clear();
    assert!(store
        .admitted_tree_pair_operation(
            &rule,
            &source_id,
            source_layout,
            &destination_id,
            destination_layout,
            |_| true,
        )
        .is_none());
}

#[test]
fn clear_prevents_a_racing_old_generation_from_reinserting() {
    // What: a compiler that began before clear may finish for its caller but
    // cannot publish into the cleared Runtime generation.
    let store = Arc::new(RuntimeTreeTransformStore::with_limits(
        2,
        usize::MAX,
        usize::MAX,
    ));
    let (key, structure, _) = fixture(2);
    let next_key = key.clone();
    let next_structure = Arc::clone(&structure);
    let started = Arc::new(Barrier::new(2));
    let resume = Arc::new(Barrier::new(2));
    let worker_store = Arc::clone(&store);
    let worker_started = Arc::clone(&started);
    let worker_resume = Arc::clone(&resume);
    let worker = std::thread::spawn(move || {
        worker_store
            .get_or_compile(key, || {
                worker_started.wait();
                worker_resume.wait();
                Ok::<_, Infallible>(structure)
            })
            .unwrap()
    });

    started.wait();
    store.clear();
    resume.wait();
    let returned = worker.join().unwrap();

    assert_eq!(returned.block_count(), 1);
    let cleared = store.info();
    assert_eq!(cleared.entries(), 0);
    assert_eq!(cleared.misses(), 0);
    store
        .get_or_compile(next_key, || Ok::<_, Infallible>(next_structure))
        .unwrap();
    let admitted = store.info();
    assert_eq!(admitted.entries(), 1);
    assert_eq!(admitted.misses(), 1);
    assert_eq!(admitted.hits(), 0);
}

#[test]
fn checked_generic_lookup_survives_core_interner_reset() {
    // What: Runtime ownership, not the process-global interner lifetime,
    // controls a completed Generic structure's semantic reuse.
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let store = RuntimeTreeTransformStore::default();
    let (key, structure, original) = fixture(31);
    store
        .get_or_compile(key, || Ok::<_, Infallible>(structure))
        .unwrap();
    let original_id = original.content_id();

    tenet_core::reset_core_intern_tables();
    let rebuilt = BlockStructure::from_blocks_with_rank(
        1,
        vec![BlockSpec::with_key(BlockKey::ordinal(31), vec![1], vec![1], 0).unwrap()],
    )
    .unwrap();
    assert_ne!(rebuilt.content_id(), original_id);

    let (cached, _) = store
        .lookup_checked_generic(
            RuleIdentity::of_type::<TestRuleIdentity>(),
            &TreeTransformOperation::permute([31], []),
            &rebuilt,
            &rebuilt,
            None,
            false,
        )
        .unwrap();
    assert!(cached.is_some());
    assert_eq!(store.info().hits(), 1);
    assert_eq!(store.info().misses(), 1);
}

#[test]
fn checked_generic_lookup_matches_fresh_logical_content_after_interner_reset() {
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let make_structure = |tag| {
        BlockStructure::from_blocks_with_rank(
            1,
            vec![BlockSpec::with_key(BlockKey::ordinal(tag), vec![1], vec![1], 0).unwrap()],
        )
        .unwrap()
    };
    let physical = make_structure(41);
    let logical = make_structure(42);
    let operation = TreeTransformOperation::permute([0], []);
    let replay = Arc::new(
        TreeTransformStructure::compile_structures(
            &physical,
            &physical,
            &[TreeTransformBlockSpec::single(0, 0, 1.0)],
        )
        .unwrap(),
    );
    let store = RuntimeTreeTransformStore::default();
    store
        .admit_checked_generic(
            RuleIdentity::of_type::<TestRuleIdentity>(),
            &operation,
            &physical,
            &physical,
            Some(&logical),
            true,
            replay,
            0,
        )
        .unwrap();
    let old_logical_id = logical.content_id();

    tenet_core::reset_core_intern_tables();
    let rebuilt_physical = make_structure(41);
    let rebuilt_logical = make_structure(42);
    assert_ne!(rebuilt_logical.content_id(), old_logical_id);

    let (cached, _) = store
        .lookup_checked_generic(
            RuleIdentity::of_type::<TestRuleIdentity>(),
            &operation,
            &rebuilt_physical,
            &rebuilt_physical,
            Some(&rebuilt_logical),
            true,
        )
        .unwrap();
    assert!(cached.is_some());
    assert_eq!(store.info().hits(), 1);
    assert_eq!(store.info().misses(), 0);
}

#[test]
fn checked_generic_cache_charges_distinct_logical_content_once() {
    let structure = |tag| {
        BlockStructure::from_blocks_with_rank(
            1,
            vec![BlockSpec::with_key(BlockKey::ordinal(tag), vec![1], vec![1], 0).unwrap()],
        )
        .unwrap()
    };
    let physical = structure(51);
    let logical = structure(52);
    let operation = TreeTransformOperation::permute([0], []);
    let replay = Arc::new(
        TreeTransformStructure::compile_structures(
            &physical,
            &physical,
            &[TreeTransformBlockSpec::single(0, 0, 1.0)],
        )
        .unwrap(),
    );
    let admit = |logical_source| {
        let store = RuntimeTreeTransformStore::default();
        store
            .admit_checked_generic(
                RuleIdentity::of_type::<TestRuleIdentity>(),
                &operation,
                &physical,
                &physical,
                logical_source,
                logical_source.is_some(),
                Arc::clone(&replay),
                0,
            )
            .unwrap();
        store.info().charged_payload_bytes()
    };
    let direct_bytes = admit(None);
    let adjoint_bytes = admit(Some(&logical));
    let logical_bytes = crate::cache::BlockStructureCacheKey::from_structure(&logical)
        .unwrap()
        .charged_retained_bytes();

    assert_eq!(adjoint_bytes - direct_bytes, logical_bytes);
}

fn plan_key(tag: usize, degeneracy: usize) -> super::CategoricalTransformKey {
    let block = BlockSpec::with_key(BlockKey::ordinal(tag), vec![degeneracy], vec![1], 0).unwrap();
    let structure = BlockStructure::from_blocks_with_rank(1, vec![block]).unwrap();
    super::CategoricalTransformKey::new(
        RuleIdentity::of_type::<TestRuleIdentity>(),
        &TreeTransformOperation::permute([0], []),
        tenet_core::FusionTreePairOrientation::Direct,
        false,
        &structure,
        &structure,
        None,
    )
}

fn empty_plan(
    _: &super::GroupSpecReuse<'_, f64>,
) -> Result<crate::TreeTransformGroupPlan<f64>, Infallible> {
    Ok(crate::TreeTransformGroupPlan::new(Vec::new()))
}

#[test]
fn plan_key_ignores_degeneracies_and_splits_on_block_keys() {
    // What: the categorical key is equal (and hashes equal) across a
    // degeneracy change and differs when the block-key set changes.
    use std::hash::BuildHasher;
    let hash = |key: &super::CategoricalTransformKey| rustc_hash::FxBuildHasher.hash_one(key);
    let (narrow, wide, other) = (plan_key(0, 1), plan_key(0, 3), plan_key(1, 1));
    assert_eq!(narrow, wide);
    assert_eq!(hash(&narrow), hash(&wide));
    assert_ne!(narrow, other);
}

#[test]
fn plan_tier_enforces_limits_counts_rebuilds_and_clears() {
    // What: the plan tier obeys its entry cap and per-entry byte cap,
    // counts a miss per rebuild, and clear releases every charge.
    let ledger = Arc::new(RuntimeTreeTransformCacheLedger::with_limits(1, usize::MAX));
    let real = RuntimeTreeTransformStore::<f64>::with_runtime_ledger(Arc::clone(&ledger));
    let complex = RuntimeTreeTransformStore::<Complex64>::with_runtime_ledger(Arc::clone(&ledger));
    real.get_or_build_plan(plan_key(0, 1), empty_plan).unwrap();
    real.get_or_build_plan(plan_key(0, 2), empty_plan).unwrap();
    real.get_or_build_plan(plan_key(1, 1), empty_plan).unwrap();
    let info = real.plan_info();
    assert_eq!((info.hits(), info.misses()), (1, 2));
    assert_eq!((info.entries(), info.evictions()), (1, 1));
    assert!(info.charged_payload_bytes() > 0);
    // The Runtime-wide ledger caps both dtypes together: a full ledger
    // makes the other dtype bypass rather than evict a foreign entry.
    complex
        .get_or_build_plan(plan_key(2, 1), |_| {
            Ok::<_, Infallible>(crate::TreeTransformGroupPlan::new(Vec::new()))
        })
        .unwrap();
    assert_eq!(ledger.plan_pair_info(&real, &complex).entries(), 1);
    assert_eq!(complex.plan_info().admission_bypasses(), 1);
    // Structure accounting is untouched by plan admissions.
    assert_eq!(ledger.store_pair_info(&real, &complex).entries(), 0);

    real.clear();
    complex.clear();
    let cleared = ledger.plan_pair_info(&real, &complex);
    assert_eq!(cleared.entries(), 0);
    assert_eq!(cleared.charged_payload_bytes(), 0);
    assert_eq!((cleared.hits(), cleared.misses()), (0, 0));

    let oversized = RuntimeTreeTransformStore::<f64>::with_limits(2, usize::MAX, 1);
    oversized
        .get_or_build_plan(plan_key(0, 1), empty_plan)
        .unwrap();
    assert_eq!(oversized.plan_info().entries(), 0);
    assert_eq!(oversized.plan_info().admission_bypasses(), 1);
}

#[test]
fn plan_charge_covers_spare_spec_capacity() {
    // What: the charge bounds allocated spec slots, not only used ones.
    let spec = core::mem::size_of::<crate::TreeTransformGroupBlockSpec<f64>>();
    let tight = super::charged_plan_bytes(&crate::TreeTransformGroupPlan::<f64>::new(Vec::new()));
    let slack = super::charged_plan_bytes(&crate::TreeTransformGroupPlan::<f64>::new(
        Vec::with_capacity(8),
    ));
    assert!(slack >= tight + 8 * spec);
}

fn pair_key(codomain: [usize; 2]) -> tenet_core::FusionTreePairKey {
    tenet_core::FusionTreePairKey::try_pair_from_sector_ids(
        codomain,
        [1],
        1,
        [false, false],
        [false],
        [],
        [],
        [1],
        [],
    )
    .unwrap()
}

fn one_spec(key: &tenet_core::FusionTreePairKey) -> Arc<[crate::TreeTransformGroupBlockSpec<f64>]> {
    Arc::from(vec![crate::TreeTransformGroupBlockSpec::single(
        key.clone(),
        key.clone(),
        1.0,
    )])
}

#[test]
fn group_tier_hits_equal_ordered_trees_and_obeys_limits() {
    // What: a borrowed group lookup hits an admitted group with the same
    // external sectors and ordered trees, misses a reordered tree list,
    // counts one miss per group to build, and obeys the entry cap.
    let ledger = Arc::new(RuntimeTreeTransformCacheLedger::with_limits(1, usize::MAX));
    let store = RuntimeTreeTransformStore::<f64>::with_runtime_ledger(Arc::clone(&ledger));
    let (first, second) = (pair_key([1, 0]), pair_key([0, 1]));
    let group_key = first.group_key();
    store
        .get_or_build_plan(plan_key(0, 1), |reuse| {
            let ordered: &[&tenet_core::FusionTreePairKey] = &[&first, &second];
            let reordered: &[&tenet_core::FusionTreePairKey] = &[&second, &first];
            let [super::GroupSlot::Miss(hash)] = reuse.lookup(&[(&group_key, ordered)])[..] else {
                panic!("an empty tier must miss");
            };
            reuse.admit(vec![(hash, (&group_key, ordered), one_spec(&first))]);
            let slots = reuse.lookup(&[(&group_key, ordered), (&group_key, reordered)]);
            assert!(matches!(slots[0], super::GroupSlot::Hit(_)));
            assert!(matches!(slots[1], super::GroupSlot::Miss(_)));
            let super::GroupSlot::Miss(hash) = slots[1] else {
                unreachable!()
            };
            reuse.admit(vec![(hash, (&group_key, reordered), one_spec(&second))]);
            empty_plan(reuse)
        })
        .unwrap();
    let info = store.group_info();
    assert_eq!((info.hits(), info.misses()), (1, 2));
    assert_eq!((info.entries(), info.evictions()), (1, 1));
    assert!(info.charged_payload_bytes() > 0);
    assert_eq!(ledger.group_pair_info(&store, &store).entries(), 1);
    // Group admissions charge only the group account.
    assert_eq!(ledger.store_pair_info(&store, &store).entries(), 0);

    store.clear();
    let cleared = store.group_info();
    assert_eq!((cleared.entries(), cleared.charged_payload_bytes()), (0, 0));
    assert_eq!((cleared.hits(), cleared.misses()), (0, 0));
}

#[test]
fn group_built_across_a_clear_is_not_admitted() {
    // What: groups built before a racing clear are not published into the
    // cleared generation.
    let store = RuntimeTreeTransformStore::<f64>::with_limits(2, usize::MAX, usize::MAX);
    let key = pair_key([1, 0]);
    let group_key = key.group_key();
    store
        .get_or_build_plan(plan_key(0, 1), |reuse| {
            let keys: &[&tenet_core::FusionTreePairKey] = &[&key];
            let [super::GroupSlot::Miss(hash)] = reuse.lookup(&[(&group_key, keys)])[..] else {
                panic!("an empty tier must miss");
            };
            store.clear();
            reuse.admit(vec![(hash, (&group_key, keys), one_spec(&key))]);
            empty_plan(reuse)
        })
        .unwrap();
    assert_eq!(store.group_info().entries(), 0);
}

#[test]
fn plan_built_across_a_clear_is_not_admitted() {
    // What: a plan whose build raced a clear is returned to its caller but
    // not published into the cleared generation.
    let store = RuntimeTreeTransformStore::<f64>::with_limits(2, usize::MAX, usize::MAX);
    store
        .get_or_build_plan(plan_key(0, 1), |reuse| {
            store.clear();
            empty_plan(reuse)
        })
        .unwrap();
    assert_eq!(store.plan_info().entries(), 0);
}
