use super::*;

fn isolated(name: &str) -> bool {
    const ENV: &str = "TENET_STAGED_CONTENT_ISOLATED";
    if std::env::var_os(ENV).is_some() {
        return true;
    }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", name, "--nocapture"])
        .env(ENV, "1")
        .output()
        .unwrap();
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("1 passed"),
        "isolated test must execute exactly one case: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert!(
        output.status.success(),
        "isolated staged-content test failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    false
}

fn prepared() -> PreparedBlockStructure {
    prepared_with_backing().0
}

fn prepared_with_backing() -> (PreparedBlockStructure, usize) {
    let degeneracy =
        DegeneracyStructure::packed_column_major(3, (0..4).map(|_| vec![2, 3, 4])).unwrap();
    let backing = degeneracy.blocks().next().unwrap().shape().as_ptr() as usize;
    let prepared = PreparedBlockStructure::from_parts(
        SectorStructure::from_keys(3, (0..4).map(BlockKey::ordinal)).unwrap(),
        degeneracy,
    )
    .unwrap();
    (prepared, backing)
}

#[test]
fn preview_commit_reuses_one_unpublished_content() {
    if !isolated(
        "tests::block_structure::staged_content::preview_commit_reuses_one_unpublished_content",
    ) {
        return;
    }
    reset_core_intern_tables();
    let (staged, backing) = prepared_with_backing();
    assert_eq!(
        staged.structure().block(0).unwrap().shape().as_ptr() as usize,
        backing
    );
    let retained_preview = staged.structure().clone();
    let preview = retained_preview.content_key();
    assert!(Arc::ptr_eq(&preview, &staged.structure().content_key()));
    assert_eq!(block_structure_intern_cache_info().entries(), 0);
    let committed = staged.commit();
    assert!(Arc::ptr_eq(&preview, &committed.content_key()));
    assert_eq!(block_structure_intern_cache_info().entries(), 1);
}

#[test]
fn reset_after_preview_keeps_old_candidate_uncached() {
    if !isolated(
        "tests::block_structure::staged_content::reset_after_preview_keeps_old_candidate_uncached",
    ) {
        return;
    }
    let staged = prepared();
    let preview = staged.structure().content_key();
    reset_core_intern_tables();
    let committed = staged.commit();
    assert_eq!(block_structure_intern_cache_info().entries(), 0);
    assert!(Arc::ptr_eq(&preview, &committed.content_key()));
}

#[test]
fn live_canonical_winner_preserves_preview_semantics_and_tiling() {
    if !isolated("tests::block_structure::staged_content::live_canonical_winner_preserves_preview_semantics_and_tiling") {
        return;
    }
    reset_core_intern_tables();
    let winner = prepared().commit();
    let staged = prepared().with_storage_tiling();
    let preview = staged.structure().clone();
    assert!(preview.storage_tiling_proven());
    let preview_regions = preview.coupled_sector_regions(1).unwrap();
    let committed = staged.commit();
    assert!(Arc::ptr_eq(&winner.content_key(), &committed.content_key()));
    assert!(!Arc::ptr_eq(
        &preview.content_key(),
        &committed.content_key()
    ));
    assert_eq!(preview, committed);
    assert_eq!(
        preview_regions,
        committed.coupled_sector_regions(1).unwrap()
    );
    assert!(committed.storage_tiling_proven());
    assert_eq!(block_structure_intern_cache_info().entries(), 1);
}

#[test]
fn equal_previewed_commits_select_one_race_winner() {
    if !isolated(
        "tests::block_structure::staged_content::equal_previewed_commits_select_one_race_winner",
    ) {
        return;
    }
    fn send_sync<T: Send + Sync>() {}
    send_sync::<PreparedBlockStructure>();
    reset_core_intern_tables();
    let barrier = Arc::new(std::sync::Barrier::new(2));
    let threads: Vec<_> = (0..2)
        .map(|_| {
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                let staged = prepared();
                let preview = staged.structure().clone();
                barrier.wait();
                let committed = staged.commit();
                assert_eq!(preview, committed);
                (preview, committed)
            })
        })
        .collect();
    let results: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
    assert!(Arc::ptr_eq(
        &results[0].1.content_key(),
        &results[1].1.content_key()
    ));
    assert_eq!(block_structure_intern_cache_info().entries(), 1);
}

#[test]
fn reset_stale_preview_can_hit_new_winner_without_republication() {
    if !isolated("tests::block_structure::staged_content::reset_stale_preview_can_hit_new_winner_without_republication") {
        return;
    }
    let staged = prepared();
    let old = staged.structure().content_key();
    reset_core_intern_tables();
    let winner = prepared().commit();
    let before = block_structure_intern_cache_info();
    let committed = staged.commit();
    assert!(Arc::ptr_eq(&winner.content_key(), &committed.content_key()));
    assert!(!Arc::ptr_eq(&old, &committed.content_key()));
    assert_eq!(block_structure_intern_cache_info(), before);
}

#[test]
fn preview_built_during_reset_is_never_published() {
    if !isolated(
        "tests::block_structure::staged_content::preview_built_during_reset_is_never_published",
    ) {
        return;
    }
    let saved = std::rc::Rc::new(std::cell::RefCell::new(None));
    let slot = std::rc::Rc::clone(&saved);
    crate::block_structure::MID_RESET_HOOK.with(|hook| {
        hook.set(Some(Box::new(move || {
            let staged = prepared();
            let preview = staged.structure().content_key();
            *slot.borrow_mut() = Some((staged, preview));
        })));
    });
    reset_core_intern_tables();
    let (staged, preview) = saved.borrow_mut().take().unwrap();
    let committed = staged.commit();
    assert!(Arc::ptr_eq(&preview, &committed.content_key()));
    assert_eq!(block_structure_intern_cache_info().entries(), 0);
}

#[test]
fn concurrent_preview_initialization_moves_backing_once() {
    let (staged, backing) = prepared_with_backing();
    let staged = Arc::new(staged);
    let threads: Vec<_> = (0..4)
        .map(|_| {
            let staged = Arc::clone(&staged);
            std::thread::spawn(move || {
                assert_eq!(
                    staged.structure().block(0).unwrap().shape().as_ptr() as usize,
                    backing
                );
                staged.structure().content_key()
            })
        })
        .collect();
    let contents: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
    assert!(contents.iter().all(|c| Arc::ptr_eq(c, &contents[0])));
}

#[test]
fn reset_started_after_epoch_guard_clears_the_locked_publication() {
    if !isolated("tests::block_structure::staged_content::reset_started_after_epoch_guard_clears_the_locked_publication") {
        return;
    }
    reset_core_intern_tables();
    let staged = prepared();
    let preview = staged.structure().content_key();
    let reset_thread = std::rc::Rc::new(std::cell::RefCell::new(None));
    let slot = std::rc::Rc::clone(&reset_thread);
    crate::block_structure::PREPARED_INTERN_BEFORE_INSERT_HOOK.with(|hook| {
        hook.set(Some(Box::new(move || {
            let (ready, reached) = std::sync::mpsc::channel();
            *slot.borrow_mut() = Some(std::thread::spawn(move || {
                crate::block_structure::MID_RESET_HOOK.with(|hook| {
                    hook.set(Some(Box::new(move || ready.send(()).unwrap())));
                });
                reset_core_intern_tables();
            }));
            reached.recv().unwrap();
            // Reset has made the epoch odd, but cannot clear the content
            // table until this insertion releases its write lock.
            assert!(!crate::block_structure::core_reset_epoch().is_multiple_of(2));
        })));
    });
    let committed = staged.commit();
    reset_thread.borrow_mut().take().unwrap().join().unwrap();
    assert!(Arc::ptr_eq(&preview, &committed.content_key()));
    assert_eq!(block_structure_intern_cache_info().entries(), 0);
    assert_ne!(prepared().commit().content_id(), committed.content_id());
}

#[test]
fn stale_candidate_cannot_replace_a_post_reset_dead_entry() {
    if !isolated("tests::block_structure::staged_content::stale_candidate_cannot_replace_a_post_reset_dead_entry") {
        return;
    }
    let staged = prepared();
    let preview = staged.structure().content_key();
    reset_core_intern_tables();
    drop(prepared().commit());
    let before = block_structure_intern_cache_info();
    let committed = staged.commit();
    assert!(Arc::ptr_eq(&preview, &committed.content_key()));
    assert_eq!(block_structure_intern_cache_info(), before);
    assert_ne!(prepared().commit().content_id(), committed.content_id());
}

#[test]
fn fresh_preview_replaces_dead_entry_with_its_own_content() {
    if !isolated("tests::block_structure::staged_content::fresh_preview_replaces_dead_entry_with_its_own_content") {
        return;
    }
    reset_core_intern_tables();
    let first = prepared().commit();
    let old_id = first.content_id();
    drop(first);
    let staged = prepared();
    let preview = staged.structure().content_key();
    let committed = staged.commit();
    assert!(Arc::ptr_eq(&preview, &committed.content_key()));
    assert_ne!(committed.content_id(), old_id);
    assert_eq!(block_structure_intern_cache_info().entries(), 1);
}

#[test]
fn preparation_before_reset_with_first_preview_after_reset_can_publish() {
    if !isolated("tests::block_structure::staged_content::preparation_before_reset_with_first_preview_after_reset_can_publish") {
        return;
    }
    let (prepared, _) = prepared_with_backing();
    reset_core_intern_tables();
    let preview = prepared.structure().clone();
    let committed = prepared.commit();
    assert!(Arc::ptr_eq(
        &preview.content_key(),
        &committed.content_key()
    ));
    assert_eq!(block_structure_intern_cache_info().entries(), 1);
}

#[cfg(feature = "racah-generated")]
#[test]
fn canonical_winner_keeps_previewed_coupled_region_geometry() {
    if !isolated("tests::block_structure::staged_content::canonical_winner_keeps_previewed_coupled_region_geometry") {
        return;
    }
    reset_core_intern_tables();
    let rule = SUNFusionRule::new(3).unwrap();
    let adjoint = rule.encode_dynkin(&[1, 1]).unwrap();
    let trivial = rule.encode_dynkin(&[0, 0]).unwrap();
    let leg = |dual| SectorLeg::new([(adjoint, 2), (trivial, 3)], dual);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(false), leg(true)]),
        FusionProductSpace::new([leg(false), leg(true)]),
    );
    let stage = || {
        hom.prepare_coupled_subblock_structure_from_leg_degeneracies_generic_checked(&rule)
            .unwrap()
    };
    let winner = stage().commit();
    let staged = stage();
    let preview = staged.structure().clone();
    let regions = preview.coupled_sector_regions(2).unwrap().unwrap();
    assert!(regions.len() > 1);
    let committed = staged.commit();
    assert!(Arc::ptr_eq(&winner.content_key(), &committed.content_key()));
    assert!(!Arc::ptr_eq(
        &preview.content_key(),
        &committed.content_key()
    ));
    assert!(Arc::ptr_eq(
        &regions,
        &committed.coupled_sector_regions(2).unwrap().unwrap()
    ));
    assert_eq!(preview, committed);
    assert!(committed.storage_tiling_proven());
}
