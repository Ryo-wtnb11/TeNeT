"""External key tracer for the complete hom-space structure census (#1365).

Reads each lookup key of `tenet_core::complete_hom_space_structure_cached`
from a debug build of `tenet-network/examples/complete_structure_census.rs`
without touching production code: lldb breaks on the lookup, on the admission
(for the charged bytes) and on the example's `census_mark` step boundary.
A key is named by its HomSpace content (per leg: duality, sector ids,
degeneracies), which is its cache identity within one rule; each workload
resets the cache and uses one rule.

    CENSUS_TRACE_OUT=trace.txt lldb -b \
        -o 'command script import benchmarks/complete_structure_census_lldb.py' \
        -o 'run > census-debug.csv' -o quit \
        target/debug/examples/complete_structure_census

Output lines: `M <step>`, `L <key>`, `A <key> <charged_bytes>`.
"""

import hashlib
import os

import lldb

OUT = None
TYPES = {}


def u64(process, addr):
    err = lldb.SBError()
    value = process.ReadUnsignedFromMemory(addr, 8, err)
    if not err.Success():
        raise RuntimeError(err.GetCString())
    return value


def field(sbtype, name):
    for i in range(sbtype.GetNumberOfFields()):
        f = sbtype.GetFieldAtIndex(i)
        if f.GetName() == name:
            return f.GetOffsetInBytes(), f.GetType()
    raise KeyError(name)


def smallvec(process, addr, svtype, elem_size):
    """(pointer, len) of a `smallvec::SmallVec` (union layout) at `addr`."""
    cap_off, _ = field(svtype, "capacity")
    data_off, data_type = field(svtype, "data")
    inline = data_type.GetByteSize() // elem_size
    cap = u64(process, addr + cap_off)
    if cap <= inline:
        return addr + data_off, cap
    return u64(process, addr + data_off), u64(process, addr + data_off + 8)


def leg_words(process, leg_addr):
    t = TYPES
    data_off, _ = field(t["leg"], "data")
    dual_off, _ = field(t["leg"], "is_dual")
    err = lldb.SBError()
    dual = process.ReadUnsignedFromMemory(leg_addr + dual_off, 1, err)
    inner = u64(process, leg_addr + data_off) + 16  # ArcInner: strong, weak, data
    words = [dual]
    for name in ("sectors", "degeneracies"):
        off, svtype = field(t["legdata"], name)
        ptr, n = smallvec(process, inner + off, svtype, 8)
        words.append(n)
        words.extend(u64(process, ptr + 8 * i) for i in range(n))
    return words


def key_name(frame, key):
    process = frame.GetThread().GetProcess()
    content = key.GetChildMemberWithName("homspace").GetChildMemberWithName("ptr")
    content = content.GetChildMemberWithName("pointer").GetValueAsUnsigned() + 16
    words = []
    for side in ("codomain", "domain"):
        off, _ = field(TYPES["content"], side)
        legs_off, legs_type = field(TYPES["product"], "legs")
        ptr, n = smallvec(process, content + off + legs_off, legs_type, TYPES["leg"].GetByteSize())
        words.append(n)
        for i in range(n):
            words.extend(leg_words(process, ptr + i * TYPES["leg"].GetByteSize()))
    return hashlib.blake2b(repr(words).encode(), digest_size=8).hexdigest()


def on_mark(frame, bp_loc, extra, internal_dict):
    OUT.write(f"M {frame.FindVariable('step').GetValueAsUnsigned()}\n")
    return False


def on_lookup(frame, bp_loc, extra, internal_dict):
    OUT.write(f"L {key_name(frame, frame.FindVariable('key').Dereference())}\n")
    return False


def on_admit(frame, bp_loc, extra, internal_dict):
    charged = frame.FindVariable("charged_bytes").GetValueAsUnsigned()
    key = frame.FindVariable("key").GetChildMemberWithName("ptr")
    key = key.GetChildMemberWithName("pointer").Dereference().GetChildMemberWithName("data")
    OUT.write(f"A {key_name(frame, key)} {charged}\n")
    return False


def __lldb_init_module(debugger, internal_dict):
    global OUT
    # Line-buffered: lldb exits with the process and never closes this file.
    OUT = open(os.environ.get("CENSUS_TRACE_OUT", "census-trace.txt"), "w", buffering=1)
    target = debugger.GetSelectedTarget()
    for alias, name in (
        ("leg", "tenet_core::SectorLeg"),
        ("legdata", "tenet_core::SectorLegData"),
        ("content", "tenet_core::FusionTreeHomSpaceContent"),
        ("product", "tenet_core::FusionProductSpace"),
    ):
        TYPES[alias] = target.FindFirstType(name)
        assert TYPES[alias].IsValid(), name
    module = __name__
    bp = target.BreakpointCreateByName("complete_structure_census::census_mark")
    bp.SetScriptCallbackFunction(f"{module}.on_mark")
    bp = target.BreakpointCreateByName("tenet_core::complete_hom_space_structure_cached")
    bp.SetScriptCallbackFunction(f"{module}.on_lookup")
    bp = target.BreakpointCreateByName("tenet_core::CompleteHomSpaceStructureCache::admit_built")
    bp.SetScriptCallbackFunction(f"{module}.on_admit")
