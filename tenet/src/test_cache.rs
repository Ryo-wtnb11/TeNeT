//! Test support for the process-global structure caches.

use crate::cache::{StructureCacheInfo, StructureCacheKind};

/// The process-global completed-transformer cache.
pub(crate) fn completed() -> StructureCacheInfo {
    crate::cache::stats()
        .into_iter()
        .find(|info| info.kind() == StructureCacheKind::CompletedTreeTransformer)
        .expect("every structure cache kind reports")
}

/// Re-executes exactly one test, alone, in a child process: for assertions
/// on absolute process-global cache counters, which any concurrently running
/// sibling test moves. Call first in the test with a name-unique env var and
/// the test's libtest path; when it returns `true` the child ran the body and
/// the caller returns.
///
/// The child always runs with `--include-ignored`: the parent runs only when
/// libtest selected it, so an `#[ignore]` device test the parent reached
/// with `--ignored` must run in the child too, not be skipped by it.
pub(crate) fn run_isolated_or_return(isolated_env: &str, test_path: &str) -> bool {
    if std::env::var_os(isolated_env).is_some() {
        return false;
    }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test_path, "--include-ignored"])
        .env(isolated_env, "1")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success() && stdout.contains("test result: ok. 1 passed; 0 failed;"),
        "isolated test did not execute exactly once: {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status
    );
    true
}

#[cfg(test)]
mod tests {
    /// Stands in for an `#[ignore]` device test that isolates itself.
    #[test]
    #[ignore = "run through ignored_isolated_tests_execute_in_their_child"]
    fn ignored_isolated_probe() {
        if super::run_isolated_or_return(
            "TENET_TEST_CACHE_IGNORED_PROBE",
            "test_cache::tests::ignored_isolated_probe",
        ) {
            return;
        }
        assert!(std::env::var_os("TENET_TEST_CACHE_IGNORED_PROBE").is_some());
    }

    #[test]
    fn ignored_isolated_tests_execute_in_their_child() {
        // What: an ignored test run with `--ignored` re-runs its body in the
        // isolated child exactly once instead of failing the helper's
        // "exactly once" check because the child skipped it.
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "test_cache::tests::ignored_isolated_probe",
                "--ignored",
            ])
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success() && stdout.contains("test result: ok. 1 passed; 0 failed;"),
            "{stdout}\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
