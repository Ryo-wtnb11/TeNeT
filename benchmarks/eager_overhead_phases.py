#!/usr/bin/env python3
"""Fold a macOS `sample` call tree of `eager_overhead_ledger` into phases.

usage: eager_overhead_phases.py SAMPLE_FILE [median_ns]

Only main-thread samples under `sample_calls` (the timed loop) are counted.
Each sample is charged to the phase of the deepest frame on its stack that
matches a phase rule; a sample with no phase frame below the public call is
charged to `facade+inlined loops` (or to `output alloc` / `drop` when its leaf is the
allocator or a destructor called directly from the facade). Rules are tried
in order per frame, so a frame matching several rules takes the first.
Owners are the outermost frame of the run of same-phase frames that contains
the deciding frame. Worker-thread samples are reported separately as busy or
idle to show which layer, if any, runs on another thread.
"""

import re
import sys
from collections import Counter, defaultdict

RULES = [
    ("gemm/qr kernel", r"^(faer|gemm_|gemm::|nano_gemm|pulp|dyn_stack)|householder|faer::linalg"),
    ("strided copy kernel", r"apply_fused_pair_slices|strided_kernel::|strided_perm::|copy_scale_strided|kernel_adapter::"),
    ("backend session entry", r"^_?<?tenferro|tenferro_cpu|tenferro_runtime|tenferro_linalg|tenet_dense::"),
    ("workspace lease", r"[Ww]orkspace|lease|Scratch|scratch"),
    ("eager error value drop", r"drop_in_place<[^>]*Error>"),
    ("drop", r"drop_in_place|drop_slow|<.* as core::ops::drop::Drop>::drop"),
    ("key hashing", r"hashbrown|core::hash|::hash::|Hasher|hash_one|make_hash|_hash\b|::hash\b"),
    ("space/structure construction+interning",
     r"from_final_homspace|build_complete|PreparedFusionTreeLayout|DegeneracyBlock|HomSpaceDescriptor|"
     r"try_permute_checked|tensorcontract_homspace|contracted_space|contracted_multiplicity_free|"
     r"[Ii]ntern|BlockStructure|FusionTreeHomSpace|FusionProductSpace|DynamicFusionMapSpace::|"
     r"root_with_replaced_leg|checked_metadata_dispatcher|fusion_space::|block_structure::|homspace|HomSpace"),
    ("plan/cache lookup",
     r"get_or_|[Cc]ache|lookup|compile_|prepare_|_plan|Plan<|plan::|resolution::|encoded_layout_primer|"
     r"artifact"),
    ("validation", r"[Vv]alidat|[Pp]reflight|require_|check_|_checked|admission|Admission"),
    ("fused-layout/kernel dispatch",
     r"transform_replay::|fusion_replay::|replay_backend::|tree_transform|transform_structure::|"
     r"execute_|oriented_fusion_restrict|restrict_into|matrixalgebra|factor"),
    ("output alloc+zeroing", r"zeroed_payload|__rust_alloc_zeroed|alloc_zeroed|calloc"),
]
COMPILED = [(phase, re.compile(rx)) for phase, rx in RULES]
ALLOC_LEAF = re.compile(
    r"^(malloc|free|calloc|realloc|platform_mem(set|move)|bzero|szone_|nanov2_|tiny_|small_|medium_|"
    r"mach_vm|madvise|mem(cpy|move|set)|RNvCs\w*rust_(alloc|dealloc|realloc|alloc_zeroed))"
)

LINE = re.compile(r"^(\s*[+!:| ]*?)(\d+) (.*)$")


def clean(name):
    name = re.sub(r"::h[0-9a-f]{16}", "", name)
    name = re.sub(r"\s+\(in [^)]*\).*$", "", name)
    for a, b in (("$LT$", "<"), ("$GT$", ">"), ("$u20$", " "), ("$C$", ","), ("$RF$", "&"),
                 ("$u7b$", "{"), ("$u7d$", "}"), ("..", "::")):
        name = name.replace(a, b)
    return name.lstrip("_").strip()


def parse(path):
    text = open(path).read()
    graph = text.split("Call graph:")[1].split("Total number in stack")[0]
    threads = []  # (thread header, list of (path, self_count))
    stack = []  # (depth, count, name, child_sum)
    current = None

    def pop_to(depth):
        while stack and stack[-1][0] >= depth:
            d, c, n, cs = stack.pop()
            own = c - cs
            if own > 0 and current is not None:
                current[1].append(([s[2] for s in stack] + [n], own))

    for line in graph.splitlines():
        m = LINE.match(line)
        if not m:
            continue
        depth, count, name = len(m.group(1)), int(m.group(2)), m.group(3)
        if name.startswith("Thread_"):
            pop_to(0)
            current = (name, [])
            threads.append(current)
            stack.clear()
            continue
        pop_to(depth)
        if stack:
            stack[-1] = (stack[-1][0], stack[-1][1], stack[-1][2], stack[-1][3] + count)
        stack.append((depth, count, clean(name), 0))
    pop_to(0)
    return threads


def classify(frames):
    start = None
    for i, f in enumerate(frames):
        if f.startswith("eager_overhead_ledger::sample_calls"):
            start = i + 1
    if start is None:
        return None
    body = frames[start:]
    best = None
    for i, f in enumerate(body):
        for phase, rx in COMPILED:
            if rx.search(f):
                best = (i, phase)
                break
    if best is None:
        leaf = body[-1] if body else ""
        phase = "output alloc+zeroing" if ALLOC_LEAF.search(leaf) else "facade+inlined loops"
        owner = next((f for f in reversed(body) if f.startswith("tenet")), leaf)
        return phase, owner
    i, phase = best
    owner = body[i]
    for j in range(i - 1, -1, -1):
        hit = next((p for p, rx in COMPILED if rx.search(body[j])), None)
        if hit == phase:
            owner = body[j]
        elif hit is not None:
            break
    return phase, owner


def main():
    path = sys.argv[1]
    median_ns = float(sys.argv[2]) if len(sys.argv) > 2 else None
    threads = parse(path)
    phases, owners, alloc_leaf = Counter(), defaultdict(Counter), Counter()
    workers = Counter()
    for header, samples in threads:
        main_thread = "main-thread" in header
        for frames, count in samples:
            if main_thread:
                got = classify(frames)
                if got is None:
                    continue
                phase, owner = got
                phases[phase] += count
                owners[phase][owner] += count
                if ALLOC_LEAF.search(frames[-1]):
                    alloc_leaf[phase] += count
            else:
                idle = re.search(r"cvwait|psynch|Sleep::sleep|wait_until|__workq_kernreturn|mach_msg|semwait|kevent", frames[-1] + " ".join(frames[-3:]))
                layer = "rayon" if any("rayon" in f for f in frames) else "other"
                workers[(layer, "idle" if idle else "busy")] += count
    total = sum(phases.values())
    print(f"# {path}: {total} main-thread samples in the timed loop")
    print("| phase | share | ns/call | of which allocator/memcpy leaf | top owners |")
    print("|---|---:|---:|---:|---|")
    for phase, count in phases.most_common():
        share = count / total
        ns = f"{share * median_ns:.0f}" if median_ns else ""
        top = "; ".join(f"`{o[:90]}` {c / total:.0%}" for o, c in owners[phase].most_common(3))
        print(f"| {phase} | {share:.1%} | {ns} | {alloc_leaf[phase] / total:.1%} | {top} |")
    if workers:
        print("worker threads (samples):", dict(workers))


if __name__ == "__main__":
    main()
