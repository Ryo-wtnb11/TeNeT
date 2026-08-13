import subprocess
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "repository-rules-review.py"
SHA = "a" * 40
DIFF = "diff --git a/README.md b/README.md\n+++ b/README.md\n+backend cuda cache oracle"


def run(*args, input_text=None):
    return subprocess.run([sys.executable, str(SCRIPT), *args], input=input_text,
                          text=True, capture_output=True)


class RepositoryRulesReviewTests(unittest.TestCase):
  def test_routes_in_stable_order_and_prints_exact_commit(self):
    result = run("--commit", SHA, "--diff", "-", input_text=DIFF)
    self.assertEqual(result.returncode, 0)
    self.assertEqual(result.stdout.splitlines(), [
        f"commit: {SHA}", "## capability/docs", "## backend/device",
        "## provider/oracle", "## cache/performance",
    ])

  def test_unknown_patch_uses_fallback_lane(self):
    result = run("--commit", SHA, "--diff", "-", input_text="diff --git a/x b/x\n+x")
    self.assertEqual(result.returncode, 0)
    self.assertEqual(result.stdout.splitlines(), [f"commit: {SHA}", "## fallback"])

  def test_rejects_short_commit_and_malformed_diff(self):
    self.assertNotEqual(run("--commit", "abc", "--diff", "-", input_text=DIFF).returncode, 0)
    result = run("--commit", SHA, "--diff", "-", input_text="not a diff")
    self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
