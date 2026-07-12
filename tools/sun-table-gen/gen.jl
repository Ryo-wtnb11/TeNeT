# SU(N) fusion-symbol table generator (Stage B3b/B3c, tenet #97/#113).
#
# Emits a group-agnostic tabulated fusion-symbol blob loaded by the Rust
# `TabulatedFusionRule` provider (tenet-core/src/su3.rs) via `include_bytes!`
# (the checked-in SU(3) table) or `from_bytes` (any other group, e.g. the
# SU(4) smoke table). The N-/F-/R-/dim/dual/Frobenius–Schur values come
# straight from SUNRepresentations' implementation of the TensorKitSectors
# sector interface for `SUNIrrep{N}` — they are the reference, never hand-typed.
#
# N-PARAMETRIC: the only group-specific inputs are `N` and the `dim` cut; every
# fusion/F/R/frontier computation goes through the group-agnostic sector
# interface. A new group is DATA ONLY — no Rust changes.
#
#   julia --project=<combenv> gen.jl [N] [dim_cut] [out_path]
#   julia --project=<combenv> gen.jl 3 27                 # canonical SU(3) table
#   julia --project=<combenv> gen.jl 4 20 su4_table.bin   # small SU(4) smoke table
#
# Defaults: N=3, dim_cut=27, out = tenet-core/src/su3_table.bin.
#
# SectorId encoding is DENSE: id = position in the irrep list sorted by
# (dim, dynkin_label...). Vacuum (all-zero label) sorts first → id 0
# (matches FusionRule::vacuum).
#
# ROW-MAJOR OUTPUT (important): Julia arrays are column-major. A column-major
# transcription bug bit Stage B2b, so the transpose is handled HERE — every F/R
# block is flattened row-major to match the Rust reader's `GenericFArray::get` /
# `GenericRMatrix::get` indexing. The reader copies bytes verbatim and must NOT
# re-transpose.
#
# Binary format v3 (little-endian). `rank = N - 1` Dynkin labels per irrep. A
# trailing FNV-1a-64 of the whole payload is stored in the header so the Rust
# loader can self-check against corruption / a transpose mistake:
#   magic         : 4  bytes = b"TFR3"
#   version       : u32 = 3
#   group_n       : u32   (N of SU(N); rank = N-1 = labels per irrep)
#   provenance    : u64   (FNV-1a-64 of every byte after this field)
#   n_irreps      : u32
#   irreps        : n_irreps × { label: rank×u8, dim:u32, dual:u8, fs:i8 }
#   n_pairs       : u32   (COVERED pairs only: all channels in-set)
#   pairs         : n_pairs × { a:u8, b:u8, n_ch:u8, [c:u8, N:u8]×n_ch }
#   n_r           : u32
#   r             : n_r × { a:u8,b:u8,c:u8, rows:u8, cols:u8, [f64]×rows*cols }
#   n_f           : u32
#   f             : n_f × { a,b,c,d,e,f:u8, s0,s1,s2,s3:u8, [f64]×∏s }
# --- frontier shell (integers only, no frontier F/R symbols) ---
#   n_frontier    : u32   (out-of-table channels of in-table pairs)
#   frontier      : n × { label: rank×u8, dim:u32, dual_fid:u16 }
#   n_escaping    : u32   (in-table pairs with >=1 out-of-table channel)
#   escaping      : n × { a:u8, b:u8, n_in:u8, [c:u8,N:u8]×, n_fr:u8, [fid:u16]× }
#   n_hops        : u32   (one-hop returns: frontier f ⊗ in-table x)
#   hops          : n × { fid:u16, x:u8, flags:u8, n_ret:u8, [c:u8,N:u8]× }
#                   flags bit0 = f⊗x has channels beyond table ∪ frontier
#                   flags bit1 = f⊗x has frontier (first-shell) channels
# (sector ids are u8: small tables have <256 in-set irreps; frontier ids are
#  u16 indices into the frontier list. The shell lets the Rust coupled-sector
#  fold classify sectors as clean / tainted / escaped instead of panicking —
#  see su3.rs `coupled_sector_fold`.)

using LinearAlgebra, SUNRepresentations, TensorKitSectors

const N       = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 3
const DIM_CUT = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 27
const RANK    = N - 1
const DEST    = length(ARGS) >= 3 ? ARGS[3] :
    joinpath(@__DIR__, "..", "..", "tenet-core", "src", "su3_table.bin")

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
pu16!(v, x) = for shift in (0, 8); pu8!(v, (UInt16(x) >> shift) & 0xff); end
pu32!(v, x) = for shift in (0, 8, 16, 24); pu8!(v, (UInt32(x) >> shift) & 0xff); end
pu64!(v, x) = for shift in (0, 8, 16, 24, 32, 40, 48, 56); pu8!(v, (UInt64(x) >> shift) & 0xff); end
pf64!(v, x) = pu64!(v, reinterpret(UInt64, Float64(x)))

# label as a RANK-tuple of ints (group-agnostic Dynkin coordinates)
label_of(s) = collect(Int, dynkin_label(s))
plabel!(v, s) = for c in label_of(s); pu8!(v, c); end

function main()
    Irr = SUNIrrep{N}

    # ---- irrep set (dim <= cut), dense id = sorted position ----
    # Each Dynkin component p_i gives dim >= p_i + 1, so p_i <= DIM_CUT - 1 is a
    # safe (loose) bound; the dim filter does the real cut.
    irreps = Irr[]
    maxc = DIM_CUT - 1
    for coords in Iterators.product(ntuple(_ -> 0:maxc, RANK)...)
        s = Irr(UInt8.(coords))
        dim(s) <= DIM_CUT && push!(irreps, s)
    end
    sort!(irreps, by = s -> (dim(s), label_of(s)...))
    id = Dict(s => i - 1 for (i, s) in enumerate(irreps))
    inset(s) = haskey(id, s)
    n = length(irreps)
    @assert all(==(0), label_of(irreps[1])) "vacuum must be id 0"

    payload = UInt8[]

    # ---- irreps ----
    pu32!(payload, n)
    for s in irreps
        plabel!(payload, s)
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
        @assert (rows, cols) == (Nsymbol(irreps[a + 1], irreps[b + 1], irreps[c + 1]),
                                 Nsymbol(irreps[b + 1], irreps[a + 1], irreps[c + 1]))
        @assert isapprox(R' * R, I; atol = 1.0e-12, rtol = 1.0e-12)
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
                    push!(frecs, ((id[a], id[b], id[c], id[d], id[e], id[f], size(F)...), F))
                    nf += 1
                end
            end
        end
    end
    pu32!(payload, nf)
    for (hdr, F) in frecs
        a, b, c, d, e, f, s0, s1, s2, s3 = hdr
        @assert (s0, s1, s2, s3) == (
            Nsymbol(irreps[a + 1], irreps[b + 1], irreps[e + 1]),
            Nsymbol(irreps[e + 1], irreps[c + 1], irreps[d + 1]),
            Nsymbol(irreps[b + 1], irreps[c + 1], irreps[f + 1]),
            Nsymbol(irreps[a + 1], irreps[f + 1], irreps[d + 1]))
        pu8!(payload, a); pu8!(payload, b); pu8!(payload, c)
        pu8!(payload, d); pu8!(payload, e); pu8!(payload, f)
        pu8!(payload, s0); pu8!(payload, s1); pu8!(payload, s2); pu8!(payload, s3)
        for i0 in 1:s0, i1 in 1:s1, i2 in 1:s2, i3 in 1:s3   # ROW-major [mu,nu,kappa,lambda]
            pf64!(payload, F[i0, i1, i2, i3])
        end
    end

    # ---- frontier shell (Option A escape classification) ------------------
    # Frontier = every out-of-table channel of an in-table pair. Integers only:
    # no frontier F/R symbols exist, which is exactly why returned-through-
    # frontier sectors are Err, not enumerable.
    frontier_set = Set{Irr}()
    for a in irreps, b in irreps, (c, _) in directproduct(a, b)
        inset(c) || push!(frontier_set, c)
    end
    frontier = sort!(collect(frontier_set), by = s -> (dim(s), label_of(s)...))
    fid = Dict(f => i - 1 for (i, f) in enumerate(frontier))
    shell(s) = inset(s) || haskey(fid, s)
    pu32!(payload, length(frontier))
    for f in frontier
        plabel!(payload, f)
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

    witnesses = irreps[1:min(4, length(irreps))]
    for a in witnesses, b in witnesses, c in witnesses
        @assert hexagon_equation(a, b, c; atol = 1.0e-12, rtol = 1.0e-12)
    end
    for a in witnesses, b in witnesses, c in witnesses, d in witnesses
        @assert pentagon_equation(a, b, c, d; atol = 1.0e-12, rtol = 1.0e-12)
    end

    # ---- assemble file: header + payload ----
    prov = fnv1a(payload)
    out = UInt8[]
    append!(out, Vector{UInt8}("TFR3"))
    pu32!(out, 3)          # version 3 (group-agnostic labels + group tag)
    pu32!(out, N)          # group tag: N of SU(N)
    pu64!(out, prov)       # provenance / cache-identity hash
    append!(out, payload)

    write(DEST, out)
    println("wrote ", DEST, "  (SU(", N, "), dim<=", DIM_CUT, ")")
    println("  irreps=", n, " covered_pairs=", length(pairs),
            " R=", length(rrecs), " F=", nf)
    println("  frontier=", length(frontier), " escaping_pairs=", length(escrecs),
            " one_hop=", nhops)
    println("  bytes=", length(out), " (", round(length(out) / 1e6, digits = 3), " MB)")
    println("  provenance FNV-1a-64 = 0x", string(prov, base = 16))
end

main()
