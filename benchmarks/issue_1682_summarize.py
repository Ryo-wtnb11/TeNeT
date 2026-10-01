"""Summarize the matched full-call CSVs; raw runs remain the authority."""
import csv
import math
import statistics
from pathlib import Path

root = Path(__file__).with_name("issue_1682_results")
key_fields = ["family", "dtype", "sectors", "k", "dual", "storage", "operation"]
runs = {}
for kind in ("baseline", "candidate"):
    runs[kind] = []
    for path in sorted(root.glob(f"{kind}-*.csv")):
        rows = list(csv.DictReader(path.open()))
        table = {tuple(row[k] for k in key_fields): row for row in rows}
        assert len(table) == len(rows) == 96, path
        runs[kind].append(table)
assert len(runs["baseline"]) == len(runs["candidate"]) >= 3
keys = runs["baseline"][0].keys()
assert all(table.keys() == keys for tables in runs.values() for table in tables)
summary = []
for key in keys:
    row = dict(zip(key_fields, key))
    for kind in runs:
        for field in ("median_ns", "alloc_calls", "alloc_bytes"):
            values = [int(table[key][field]) for table in runs[kind]]
            row[f"{kind}_{field}"] = statistics.median(values)
            row[f"{kind}_{field}_min"] = min(values)
            row[f"{kind}_{field}_max"] = max(values)
    row["speedup"] = row["baseline_median_ns"] / row["candidate_median_ns"]
    summary.append(row)
with (root / "summary.csv").open("w") as out:
    writer = csv.DictWriter(out, fieldnames=summary[0].keys(), lineterminator="\n")
    writer.writeheader()
    writer.writerows(summary)
for storage in ("diagonal", "dense"):
    selected = [r for r in summary if r["storage"] == storage]
    ratios = [r["speedup"] for r in selected]
    print(storage, "cases", len(selected), "speedup min/geomean/max",
          min(ratios), math.exp(statistics.mean(map(math.log, ratios))), max(ratios))
    for row in sorted(selected, key=lambda r: r["speedup"])[:5]:
        print({k: row[k] for k in key_fields + ["speedup", "baseline_median_ns",
              "candidate_median_ns", "baseline_median_ns_min", "baseline_median_ns_max",
              "candidate_median_ns_min", "candidate_median_ns_max"]})
