# Usage: python3 compare.py <this directory>. Warm cuda rows, median of the two passes per revision.
import csv, sys, math, collections, statistics
d = sys.argv[1]

def load(path):
    with open(path) as fh:
        lines = [l for l in fh if not l.startswith('#')]
    return list(csv.DictReader(lines))

KEY = ("provider","dtype","family","blocks","degeneracy","operation","target","phase")
def matrix(rev):
    runs = [{tuple(r[k] for k in KEY): r for r in load(f"{d}/{rev}-matrix-{p}.csv")} for p in (1, 2)]
    return runs
base, br = matrix("base"), matrix("branch")
keys = [k for k in base[0] if k[6] == "cuda" and k[7] == "warm" and k in br[0]]
agg = collections.defaultdict(list)
for k in keys:
    b = statistics.median(int(r[k]["ns_median"]) for r in base)
    c = statistics.median(int(r[k]["ns_median"]) for r in br)
    bh = int(base[0][k]["h2d_bytes"]); ch = int(br[0][k]["h2d_bytes"])
    ba = int(base[0][k]["alloc_bytes"]); ca = int(br[0][k]["alloc_bytes"])
    agg[(k[1], k[5])].append((c / b, b, c, bh, ch, ba, ca))
    agg[(k[1], "ALL")].append((c / b, b, c, bh, ch, ba, ca))
print("| dtype | operation | rows | median ratio br/base | geomean | slower | base ns (median) | br ns (median) | h2d B base->br | host alloc B base->br |")
print("|---|---|---|---|---|---|---|---|---|---|")
for (dt, op), v in sorted(agg.items()):
    r = [x[0] for x in v]
    g = math.exp(sum(math.log(x) for x in r) / len(r))
    print(f"| {dt} | {op} | {len(v)} | {statistics.median(r):.3f} | {g:.3f} | {sum(x > 1 for x in r)}/{len(r)} | "
          f"{statistics.median(x[1] for x in v):.0f} | {statistics.median(x[2] for x in v):.0f} | "
          f"{sum(x[3] for x in v)}->{sum(x[4] for x in v)} | {sum(x[5] for x in v)}->{sum(x[6] for x in v)} |")

print()
def sweep(rev):
    out = collections.defaultdict(list)
    for p in (1, 2):
        for r in csv.DictReader(open(f"{d}/{rev}-sweep-{p}.csv")):
            out[(r["dtype"], r["op"], int(r["bytes"]))].append((int(r["submit_ns"]), int(r["e2e_ns"])))
    return out
sb, sc = sweep("base"), sweep("branch")
print("| dtype | op | output bytes | submit base | submit br | ratio | e2e base | e2e br | ratio |")
print("|---|---|---|---|---|---|---|---|---|")
for k in sorted(sb, key=lambda k: (k[0], k[1], k[2])):
    b, c = sb[k], sc[k]
    ms = lambda v, i: statistics.median(x[i] for x in v)
    sub = f"{ms(b,0):.0f} | {ms(c,0):.0f} | {ms(c,0)/ms(b,0):.2f}" if ms(b,0) else "- | - | -"
    print(f"| {k[0]} | {k[1]} | {k[2]} | {sub} | {ms(b,1):.0f} | {ms(c,1):.0f} | {ms(c,1)/ms(b,1):.2f} |")
    print(f"<!-- passes base {b} br {c} -->")
