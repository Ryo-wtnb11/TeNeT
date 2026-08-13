#!/usr/bin/env python3
"""Route a commit diff to the repository-rule review lanes.

This is intentionally local and deterministic: it does not inspect the network
or invoke a model.  The commit is supplied by the caller so reports cannot
silently drift to another revision.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

LANES = (
    ("capability/docs", re.compile(r"(?:README|docs?/|Cargo\.toml|\.md$|capabilit|document|example)", re.I)),
    ("backend/device", re.compile(r"(?:cuda|backend|device|gpu|host|cpu|fallback)", re.I)),
    ("provider/oracle", re.compile(r"(?:racah|tensor.?kit|qspace|oracle|fusion|symmetr|provider)", re.I)),
    ("cache/performance", re.compile(r"(?:cache|benchmark|perf(?:ormance)?|allocat|memory|throughput)", re.I)),
    ("fallback", re.compile(r"(?:fallback|unsupported|error|panic|feature)", re.I)),
)


def lanes_for(diff: str) -> list[str]:
    """Return matching lanes in stable declaration order (fallback always applies)."""
    if not diff.strip() or "diff --git " not in diff:
        raise ValueError("diff input must contain a git patch")
    selected = [name for name, pattern in LANES[:-1] if pattern.search(diff)]
    if not selected:
        selected.append("fallback")
    return selected


def read_diff(value: str | None) -> str:
    if value is None or value == "-":
        return sys.stdin.read()
    try:
        return Path(value).read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read diff: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diff", help="git diff file, or '-' for stdin (default)")
    parser.add_argument("--commit", required=True, help="full 40-character commit SHA")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.commit):
        parser.error("--commit must be a full 40-character hexadecimal SHA")
    try:
        selected = lanes_for(read_diff(args.diff))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"commit: {args.commit}")
    for lane in selected:
        print(f"## {lane}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
