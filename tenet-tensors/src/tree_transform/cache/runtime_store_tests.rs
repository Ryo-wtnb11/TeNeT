use std::convert::Infallible;
use std::sync::{mpsc, Arc, Barrier};
use std::time::Duration;

use num_complex::Complex64;
use tenet_core::{BlockKey, BlockSpec, BlockStructure, RuleIdentity};

use super::{RuntimeCoefficientLedger, RuntimeCoefficientStore};
use crate::TreeTransformOperation;

struct TestRuleIdentity;

#[test]
fn reverse_pair_info_calls_complete_concurrently() {
    // What: opposite reporting order never holds both typed-store locks,
    // so it cannot form an ABBA cycle.
    let ledger = Arc::new(RuntimeCoefficientLedger::with_limits(2, usize::MAX));
    let real = Arc::new(RuntimeCoefficientStore::<f64>::with_runtime_ledger(
        Arc::clone(&ledger),
    ));
    let complex = Arc::new(RuntimeCoefficientStore::<Complex64>::with_runtime_ledger(
        Arc::clone(&ledger),
    ));
    let start = Arc::new(Barrier::new(3));
    let (done_tx, done_rx) = mpsc::channel();

    let spawn = |forward: bool, done: mpsc::Sender<()>| {
        let ledger = Arc::clone(&ledger);
        let real = Arc::clone(&real);
        let complex = Arc::clone(&complex);
        let start = Arc::clone(&start);
        std::thread::spawn(move || {
            start.wait();
            for _ in 0..5_000 {
                let _ = if forward {
                    ledger.plan_pair_info(real.as_ref(), complex.as_ref())
                } else {
                    ledger.plan_pair_info(complex.as_ref(), real.as_ref())
                };
            }
            done.send(()).unwrap();
        })
    };
    let first = spawn(true, done_tx.clone());
    let second = spawn(false, done_tx);
    start.wait();

    done_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    done_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    first.join().unwrap();
    second.join().unwrap();
}

fn plan_key(tag: usize, degeneracy: usize) -> super::CategoricalTransformKey {
    let block = BlockSpec::with_key(BlockKey::ordinal(tag), vec![degeneracy], vec![1], 0).unwrap();
    let structure = BlockStructure::from_blocks_with_rank(1, vec![block]).unwrap();
    super::CategoricalTransformKey::new(
        RuleIdentity::of_type::<TestRuleIdentity>(),
        &TreeTransformOperation::permute([0], []),
        tenet_core::FusionTreePairOrientation::Direct,
        super::OrientedBasisOrder::Canonical,
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
    let ledger = Arc::new(RuntimeCoefficientLedger::with_limits(1, usize::MAX));
    let real = RuntimeCoefficientStore::<f64>::with_runtime_ledger(Arc::clone(&ledger));
    let complex = RuntimeCoefficientStore::<Complex64>::with_runtime_ledger(Arc::clone(&ledger));
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

    real.clear();
    complex.clear();
    let cleared = ledger.plan_pair_info(&real, &complex);
    assert_eq!(cleared.entries(), 0);
    assert_eq!(cleared.charged_payload_bytes(), 0);
    assert_eq!((cleared.hits(), cleared.misses()), (0, 0));

    let oversized = RuntimeCoefficientStore::<f64>::with_limits(2, usize::MAX, 1);
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
    let ledger = Arc::new(RuntimeCoefficientLedger::with_limits(1, usize::MAX));
    let store = RuntimeCoefficientStore::<f64>::with_runtime_ledger(Arc::clone(&ledger));
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
    assert_eq!(ledger.plan_pair_info(&store, &store).entries(), 1);

    store.clear();
    let cleared = store.group_info();
    assert_eq!((cleared.entries(), cleared.charged_payload_bytes()), (0, 0));
    assert_eq!((cleared.hits(), cleared.misses()), (0, 0));
}

#[test]
fn group_built_across_a_clear_is_not_admitted() {
    // What: groups built before a racing clear are not published into the
    // cleared generation.
    let store = RuntimeCoefficientStore::<f64>::with_limits(2, usize::MAX, usize::MAX);
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
    let store = RuntimeCoefficientStore::<f64>::with_limits(2, usize::MAX, usize::MAX);
    store
        .get_or_build_plan(plan_key(0, 1), |reuse| {
            store.clear();
            empty_plan(reuse)
        })
        .unwrap();
    assert_eq!(store.plan_info().entries(), 0);
}

#[test]
fn plan_entries_charge_a_shared_content_once() {
    // What (#1998): a content retained by several plan entries is charged
    // once, and released only with its last retaining entry.
    let key = plan_key(60, 1);
    let content = RuntimeCoefficientStore::<f64>::charged_content_bytes(&key);
    let store = RuntimeCoefficientStore::<f64>::with_limits(4, usize::MAX, usize::MAX);
    store.get_or_build_plan(key.clone(), empty_plan).unwrap();
    assert_eq!(store.retained_content_bytes(), content);
    // Same sector structure, another operation: a second entry, one content.
    let block = BlockSpec::with_key(BlockKey::ordinal(60), vec![1], vec![1], 0).unwrap();
    let structure = BlockStructure::from_blocks_with_rank(1, vec![block]).unwrap();
    let transpose = super::CategoricalTransformKey::new(
        RuleIdentity::of_type::<TestRuleIdentity>(),
        &TreeTransformOperation::transpose([0], []),
        tenet_core::FusionTreePairOrientation::Direct,
        super::OrientedBasisOrder::Canonical,
        false,
        &structure,
        &structure,
        None,
    );
    store.get_or_build_plan(transpose, empty_plan).unwrap();
    assert_eq!(store.plan_info().entries(), 2);
    assert!(store.retained_content_bytes() <= 2 * content);

    store.clear();
    assert_eq!(store.retained_content_bytes(), 0);
}
