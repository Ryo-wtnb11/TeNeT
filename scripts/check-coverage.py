#!/usr/bin/env python3
"""Check per-crate line coverage floors from coverage-thresholds.json.

Usage: check-coverage.py COVERAGE_JSON [--diff-base REV]

Fails when a crate is below its floor or a floor names a crate with no
measured files. With --diff-base, also prints line coverage of the lines
added since REV; that report is informational and never fails the check.
"""

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def relative(path):
    root = str(ROOT) + "/"
    return path[len(root):] if path.startswith(root) else path


def line_hits(segments):
    """Map line -> execution count for mapped lines, following llvm-cov's
    LineCoverageStats over (line, col, count, has_count, is_entry, is_gap)."""
    by_line = defaultdict(list)
    for seg in segments:
        by_line[seg[0]].append(seg)
    hits = {}
    wrapped = None
    last = max(by_line, default=0)
    for line in range(1, last + 1):
        starts = by_line.get(line, [])
        entries = [s for s in starts if s[3] and s[4] and not s[5]]
        skipped = bool(starts) and not starts[0][3] and starts[0][4]
        if not skipped and ((wrapped and wrapped[3]) or entries):
            count = wrapped[2] if wrapped and wrapped[3] else 0
            hits[line] = max([count] + [s[2] for s in entries])
        if starts:
            wrapped = starts[-1]
    return hits


def added_lines(base):
    diff = subprocess.run(
        ["git", "diff", "-U0", "--no-color", base, "HEAD", "--", "*.rs"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout
    added = defaultdict(set)
    path = None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else None
        elif line.startswith("@@") and path:
            new = line.split()[2][1:].split(",")
            start, count = int(new[0]), int(new[1]) if len(new) > 1 else 1
            added[path].update(range(start, start + count))
    return added


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("coverage_json")
    parser.add_argument("--diff-base")
    args = parser.parse_args()

    config = json.loads((ROOT / "coverage-thresholds.json").read_text())
    default_floor = config["default"]
    floors = config["crates"]
    files = json.loads(Path(args.coverage_json).read_text())["data"][0]["files"]

    totals = defaultdict(lambda: [0, 0])
    for entry in files:
        crate = relative(entry["filename"]).split("/", 1)[0]
        lines = entry["summary"]["lines"]
        totals[crate][0] += lines["covered"]
        totals[crate][1] += lines["count"]

    failures = []
    print("Per-crate line coverage:")
    for crate, (covered, count) in sorted(totals.items()):
        percent = 100.0 * covered / count if count else 100.0
        floor = floors.get(crate, default_floor)
        status = "ok" if percent >= floor else "FAIL"
        print(f"  {crate}: {percent:.2f}% ({covered}/{count}), floor {floor}% {status}")
        if percent < floor:
            failures.append(f"{crate}: {percent:.2f}% < {floor}%")
    for crate in sorted(set(floors) - set(totals)):
        failures.append(f"{crate}: stale floor, no measured files")

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  {failure}")
    else:
        print("\nAll crates meet their coverage floors.")

    if args.diff_base:
        try:
            report_changed_lines(files, args.diff_base)
        except Exception as error:  # the report must never change the verdict
            print(f"\nChanged-line report skipped: {error}")

    sys.exit(1 if failures else 0)


def report_changed_lines(files, base):
    added = added_lines(base)
    print(f"\nChanged-line coverage since {base} (informational, never fails):")
    total_hit = total = 0
    for entry in sorted(files, key=lambda e: e["filename"]):
        path = relative(entry["filename"])
        if path not in added:
            continue
        hits = line_hits(entry["segments"])
        mapped = sorted(added[path] & hits.keys())
        missed = [n for n in mapped if hits[n] == 0]
        if not mapped:
            continue
        total += len(mapped)
        total_hit += len(mapped) - len(missed)
        print(f"  {path}: {len(mapped) - len(missed)}/{len(mapped)} added lines covered")
        if missed:
            print(f"    uncovered: {compress(missed)}")
    if total:
        print(f"  total: {total_hit}/{total} ({100.0 * total_hit / total:.1f}%)")
    else:
        print("  no added executable lines in measured files")


def compress(numbers):
    ranges, start = [], numbers[0]
    for prev, cur in zip(numbers, numbers[1:] + [None]):
        if cur != prev + 1:
            ranges.append(str(start) if start == prev else f"{start}-{prev}")
            start = cur
    return ", ".join(ranges)


if __name__ == "__main__":
    main()
