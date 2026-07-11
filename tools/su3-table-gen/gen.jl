# SU(3) fusion-symbol table generator (Stage B3b, tenet #97).
#
# Emits `tenet-core/src/su3_table.bin`, the checked-in table the Rust
# `Su3FusionRule` provider (tenet-core/src/su3.rs) loads via `include_bytes!`.
# The data is produced by SUNRepresentations' own Nsymbol/Fsymbol/Rsymbol/dim/
# dual/frobeniusschur (the TensorKitSectors sector interface) — never hand-typed
# — so the numbers are the reference, not a reproduction of it.
#
# Irrep set: every SUNIrrep{3} with dim <= 27 (17 irreps: 1,3,3̄,6,6̄,8,10,10̄,
# 15,15̄,15′,15̄′,21,21̄,24,24̄,27). This closes 8⊗8 = 1+8+8+10+10̄+27, the
# adjoint (SU(3) Heisenberg) physics motivation. Pairs whose fusion escapes the
# set (e.g. 8⊗10 ∋ 35) are the hard-error boundary: the provider panics, it
# never silently truncates.
#
# SectorId encoding is DENSE: id = position in the irrep list sorted by
# (dim, p, q). Vacuum (0,0) sorts first → id 0 (matches FusionRule::vacuum).
#
# ROW-MAJOR OUTPUT (important): Julia arrays are column-major. A column-major
# transcription bug bit Stage B2b, so the transpose is handled HERE, on the
# generator side — every F/R block is flattened row-major to match the Rust
# reader's `GenericFArray::get` / `GenericRMatrix::get` indexing. The reader does
# a byte-for-byte copy and MUST NOT re-transpose.
#
# Binary format (little-endian). A trailing FNV-1a-64 of the whole payload is
# stored in the header so the Rust loader can self-check against corruption /
# a transpose mistake:
#   magic         : 4  bytes = b"SU3T"
#   version       : u32 = 2
#   provenance    : u64  (FNV-1a-64 of every byte after this field)
#   n_irreps      : u32
#   irreps        : n_irreps × { p:u8, q:u8, dim:u32, dual:u8, fs:i8 }
#   n_pairs       : u32   (COVERED pairs only: all channels in-set)
#   pairs         : n_pairs × { a:u8, b:u8, n_ch:u8, [c:u8, n:u8]×n_ch }
#   n_r           : u32
#   r             : n_r × { a:u8,b:u8,c:u8, rows:u8, cols:u8, [f64]×rows*cols }
#   n_f           : u32
#   f             : n_f × { a,b,c,d,e,f:u8, s0,s1,s2,s3:u8, [f64]×∏s }
# --- v2 frontier shell (integers only, no frontier F/R symbols) ---
#   n_frontier    : u32   (out-of-table channels of in-table pairs)
#   frontier      : n × { p:u8, q:u8, dim:u32, dual_fid:u16 }
#   n_escaping    : u32   (in-table pairs with >=1 out-of-table channel)
#   escaping      : n × { a:u8, b:u8, n_in:u8, [c:u8,N:u8]×, n_fr:u8, [fid:u16]× }
#   n_hops        : u32   (one-hop returns: frontier f ⊗ in-table x)
#   hops          : n × { fid:u16, x:u8, flags:u8, n_ret:u8, [c:u8,N:u8]× }
#                   flags bit0 = f⊗x has channels beyond table ∪ frontier
#                   flags bit1 = f⊗x has frontier (first-shell) channels
# (sector ids are u8: only 17 in-set irreps, so 0..16 fit a byte; frontier ids
#  are u16 indices into the frontier list. The v2 shell lets the Rust-side
#  coupled-sector fold classify sectors as clean / tainted / escaped instead of
#  panicking — see su3.rs `coupled_sector_fold`.)
#
# Regenerate:  julia --project=/path/to/sunenv tools/su3-table-gen/gen.jl
# (an env with SUNRepresentations 0.4.0 + TensorKitSectors). See README.md for
# the exact provenance recorded at generation time.

using SUNRepresentations, TensorKitSectors

const DIM_CUT = 27

fnv1a(bytes::Vector{UInt8}) = begin
    h = 0xcbf29ce484222325
    for b in bytes
        h = (h ⊻ UInt64(b)) * 0x100000001b3    # wraps mod 2^64 (UInt64 arithmetic)
    end
    h
end

# little-endian appenders
pu8!(v, x)  = push!(v, UInt8(x))
pi8!(v, x)  = push!(v, reinterpret(UInt8, Int8(x)))
pu16!(v, x) = append!(v, reinterpret(UInt8, [UInt16(x)]))
pu32!(v, x) = append!(v, reinterpret(UInt8, [UInt32(x)]))
pu64!(v, x) = append!(v, reinterpret(UInt8, [UInt64(x)]))
pf64!(v, x) = append!(v, reinterpret(UInt8, [Float64(x)]))

function main()
    # ---- irrep set (dim <= 27), dense id = sorted position ----
    irreps = SUNIrrep{3}[]
    for p in 0:12, q in 0:12
        s = SUNIrrep{3}((UInt8(p), UInt8(q)))
        dim(s) <= DIM_CUT && push!(irreps, s)
    end
    sort!(irreps, by = s -> (dim(s), dynkin_label(s)[1], dynkin_label(s)[2]))
    id = Dict(s => i - 1 for (i, s) in enumerate(irreps))
    inset(s) = haskey(id, s)
    n = length(irreps)
    @assert dynkin_label(irreps[1]) == (0, 0) "vacuum must be id 0"

    payload = UInt8[]

    # ---- irreps ----
    pu32!(payload, n)
    for s in irreps
        p, q = dynkin_label(s)
        pu8!(payload, p); pu8!(payload, q)
        pu32!(payload, dim(s))
        pu8!(payload, id[dual(s)])
        pi8!(payload, round(Int, frobeniusschur(s)))
    end

    # ---- fusion: covered pairs only (every channel in-set) ----
    pairs = Vector{Tuple{Int,Int,Vector{Tuple{Int,Int}}}}()
    for a in irreps, b in irreps
        chans = collect(directproduct(a, b))          # (c => N)
        all(inset(first(cn)) for cn in chans) || continue   # escaping ⇒ hard-error, skip
        clist = [(id[c], nmul) for (c, nmul) in chans]
        sort!(clist)
        push!(pairs, (id[a], id[b], clist))
    end
    pu32!(payload, length(pairs))
    for (a, b, clist) in pairs
        pu8!(payload, a); pu8!(payload, b); pu8!(payload, length(clist))
        for (c, nmul) in clist
            pu8!(payload, c); pu8!(payload, nmul)
        end
    end

    # ---- R-symbols (all in-set triples with N>=1) ----
    rrecs = Vector{Tuple{Int,Int,Int,Matrix{Float64}}}()
    for a in irreps, b in irreps
        for (c, _) in directproduct(a, b)
            inset(c) || continue
            R = Rsymbol(a, b, c)
            push!(rrecs, (id[a], id[b], id[c], Matrix{Float64}(R)))
        end
    end
    pu32!(payload, length(rrecs))
    for (a, b, c, R) in rrecs
        rows, cols = size(R)
        pu8!(payload, a); pu8!(payload, b); pu8!(payload, c)
        pu8!(payload, rows); pu8!(payload, cols)
        for i in 1:rows, j in 1:cols          # ROW-major flatten
            pf64!(payload, R[i, j])
        end
    end

    # ---- F-symbols (all in-set 6-tuples satisfying the 4 N-conditions) ----
    nf = 0
    frecs = Vector{NTuple{2,Any}}()   # ((a,b,c,d,e,f,s...), Array)
    for a in irreps, b in irreps, c in irreps
        for (e, _) in directproduct(a, b)
            inset(e) || continue
            for (d, _) in directproduct(e, c)
                inset(d) || continue
                for (f, _) in directproduct(b, c)
                    inset(f) || continue
                    Nsymbol(a, f, d) == 0 && continue
                    F = Fsymbol(a, b, c, d, e, f)   # Array{Float64,4}
                    frecs = frecs   # keep type stable
                    push!(frecs, ((id[a], id[b], id[c], id[d], id[e], id[f], size(F)...), F))
                    nf += 1
                end
            end
        end
    end
    pu32!(payload, nf)
    for (hdr, F) in frecs
        a, b, c, d, e, f, s0, s1, s2, s3 = hdr
        pu8!(payload, a); pu8!(payload, b); pu8!(payload, c)
        pu8!(payload, d); pu8!(payload, e); pu8!(payload, f)
        pu8!(payload, s0); pu8!(payload, s1); pu8!(payload, s2); pu8!(payload, s3)
        for i0 in 1:s0, i1 in 1:s1, i2 in 1:s2, i3 in 1:s3   # ROW-major [mu,nu,kappa,lambda]
            pf64!(payload, F[i0, i1, i2, i3])
        end
    end

    # ---- v2: frontier shell (Option A escape classification) --------------
    # Frontier = every out-of-table channel of an in-table pair. Recorded so
    # the Rust coupled-sector fold can (a) keep the in-table channels of an
    # escaping pair (they are legitimate clean intermediates), (b) know WHICH
    # frontier states appear, and (c) fold a frontier state one more step via
    # the one-hop return table N(f, x, c) — enough for exact classification up
    # to one frontier hop (rank ≤ 4 single-escape); anything deeper is flagged
    # and treated conservatively (Err) on the Rust side. Integers only: no
    # frontier F/R symbols exist, which is exactly why returned-through-frontier
    # sectors are Err, not enumerable.
    frontier_set = Set{SUNIrrep{3}}()
    for a in irreps, b in irreps, (c, _) in directproduct(a, b)
        inset(c) || push!(frontier_set, c)
    end
    frontier = sort!(collect(frontier_set),
                     by = s -> (dim(s), dynkin_label(s)[1], dynkin_label(s)[2]))
    fid = Dict(f => i - 1 for (i, f) in enumerate(frontier))
    shell(s) = inset(s) || haskey(fid, s)
    pu32!(payload, length(frontier))
    for f in frontier
        p, q = dynkin_label(f)
        pu8!(payload, p); pu8!(payload, q)
        pu32!(payload, dim(f))
        pu16!(payload, fid[dual(f)])   # frontier is closed under dual
    end

    # escaping pairs: in-table channels (with N) + frontier channel ids
    escrecs = Vector{Tuple{Int,Int,Vector{Tuple{Int,Int}},Vector{Int}}}()
    for a in irreps, b in irreps
        chans = collect(directproduct(a, b))
        any(!inset(first(cn)) for cn in chans) || continue
        ins = sort!([(id[c], nmul) for (c, nmul) in chans if inset(c)])
        frs = sort!([fid[c] for (c, _) in chans if !inset(c)])
        push!(escrecs, (id[a], id[b], ins, frs))
    end
    pu32!(payload, length(escrecs))
    for (a, b, ins, frs) in escrecs
        pu8!(payload, a); pu8!(payload, b)
        pu8!(payload, length(ins))
        for (c, nmul) in ins
            pu8!(payload, c); pu8!(payload, nmul)
        end
        pu8!(payload, length(frs))
        for f in frs
            pu16!(payload, f)
        end
    end

    # one-hop returns: frontier f ⊗ in-table x
    nhops = 0
    hoprecs = Vector{Tuple{Int,Int,Int,Vector{Tuple{Int,Int}}}}()
    for f in frontier, x in irreps
        chans = collect(directproduct(f, x))
        rets = sort!([(id[c], nmul) for (c, nmul) in chans if inset(c)])
        beyond = any(!shell(first(cn)) for cn in chans)
        hasfr = any(haskey(fid, first(cn)) for cn in chans)
        flags = (beyond ? 1 : 0) | (hasfr ? 2 : 0)
        push!(hoprecs, (fid[f], id[x], flags, rets))
        nhops += 1
    end
    pu32!(payload, nhops)
    for (f, x, flags, rets) in hoprecs
        pu16!(payload, f); pu8!(payload, x); pu8!(payload, flags)
        pu8!(payload, length(rets))
        for (c, nmul) in rets
            pu8!(payload, c); pu8!(payload, nmul)
        end
    end

    # ---- assemble file: header + payload ----
    prov = fnv1a(payload)
    out = UInt8[]
    append!(out, Vector{UInt8}("SU3T"))
    pu32!(out, 2)          # version (2 = v1 + frontier shell)
    pu64!(out, prov)       # provenance / cache-identity hash
    append!(out, payload)

    dest = joinpath(@__DIR__, "..", "..", "tenet-core", "src", "su3_table.bin")
    write(dest, out)
    println("wrote ", dest)
    println("  irreps=", n, " covered_pairs=", length(pairs),
            " R=", length(rrecs), " F=", nf)
    println("  frontier=", length(frontier), " escaping_pairs=", length(escrecs),
            " one_hop=", nhops)
    println("  bytes=", length(out), " (", round(length(out) / 1e6, digits = 3), " MB)")
    println("  provenance FNV-1a-64 = 0x", string(prov, base = 16))
end

main()
