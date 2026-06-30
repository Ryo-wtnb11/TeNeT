using TensorKit

struct BenchConfig
    reps::Int
    warmup::Int
end

function parse_config()
    reps = 10_000
    warmup = 1_000
    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--reps"
            i += 1
            reps = parse(Int, ARGS[i])
        elseif ARGS[i] == "--warmup"
            i += 1
            warmup = parse(Int, ARGS[i])
        else
            error("unknown argument: $(ARGS[i])")
        end
        i += 1
    end
    return BenchConfig(reps, warmup)
end

function run_bench(f, config::BenchConfig)
    for _ in 1:config.warmup
        f()
    end
    GC.gc()
    start = time_ns()
    for _ in 1:config.reps
        f()
    end
    return time_ns() - start
end

function run_single(f)
    f()
    GC.gc()
    start = time_ns()
    result = f()
    return (time_ns() - start, result)
end

function allocated_bytes(f)
    f()
    GC.gc()
    return @allocated f()
end

function print_result(fixture, metric, ns, reps, alloc_bytes, checksum)
    avg_ns = ns / reps
    println(
        "$fixture,$metric,$reps,$(round(avg_ns; digits=3))," *
        "$alloc_bytes,$(round(real(checksum); digits=12)),$(round(imag(checksum); digits=12))",
    )
end

function deterministic_value(tensor_id, row0, col0)
    x = UInt64(tensor_id) * UInt64(1_000_003) +
        UInt64(row0) * UInt64(101) +
        UInt64(col0) * UInt64(17)
    return Float64(x % UInt64(10_000)) / 10_000 - 0.5
end

function fill_real!(t, tensor_id)
    block_id = 0
    for (_, block) in blocks(t)
        block_id += 1
        for col in axes(block, 2), row in axes(block, 1)
            block[row, col] = deterministic_value(tensor_id + block_id, row - 1, col - 1)
        end
    end
    return t
end

function fill_complex!(t, tensor_id)
    block_id = 0
    for (_, block) in blocks(t)
        block_id += 1
        for col in axes(block, 2), row in axes(block, 1)
            re = deterministic_value(tensor_id + block_id, row - 1, col - 1)
            im = deterministic_value(tensor_id + 10 + block_id, row - 1, col - 1)
            block[row, col] = complex(re, im)
        end
    end
    return t
end

function checksum(t)
    return sum(t.data)
end

function bench_su2_noncanonical_source(config::BenchConfig)
    V = SU2Space(1//2 => 2)
    A = fill_real!(zeros(Float64, (V ⊗ V ⊗ V) ← V), 11)
    B = fill_real!(zeros(Float64, V ← (V ⊗ V ⊗ V)), 12)
    C = zeros(Float64, V ← V)
    initial = [2.0, -1.0, 4.0, -3.0]
    alpha = -1.5
    beta = 0.25

    into! = () -> begin
        C.data .= initial
        @tensor C[b; a] = alpha * A[c d e; a] * B[b; c d e] + beta * C[b; a]
        return checksum(C)
    end
    alloc = () -> begin
        @tensor Out[b; a] := alpha * A[c d e; a] * B[b; c d e]
        return checksum(Out)
    end

    single_ns, single_chk = run_single(into!)
    elapsed = run_bench(into!, config)
    alloc_elapsed = run_bench(alloc, config)
    print_result("tensorkit_su2_noncanonical_source", "single_into", single_ns, 1, allocated_bytes(into!), single_chk)
    print_result("tensorkit_su2_noncanonical_source", "warm_into", elapsed, config.reps, allocated_bytes(into!), into!())
    print_result("tensorkit_su2_noncanonical_source", "alloc_contract", alloc_elapsed, config.reps, allocated_bytes(alloc), alloc())
end

function bench_su2_output_permute(config::BenchConfig)
    V = SU2Space(1//2 => 1)
    A = fill_real!(zeros(Float64, (V ⊗ V) ← (V ⊗ V)), 21)
    perm = () -> begin
        C = permute(A, ((1, 3, 2, 4), ()))
        return checksum(C)
    end
    single_ns, single_chk = run_single(perm)
    elapsed = run_bench(perm, config)
    print_result("tensorkit_su2_output_permute", "single_permute", single_ns, 1, allocated_bytes(perm), single_chk)
    print_result("tensorkit_su2_output_permute", "warm_permute", elapsed, config.reps, allocated_bytes(perm), perm())
end

function bench_product_fz2_u1_su2_complex(config::BenchConfig)
    I = FermionParity ⊠ U1Irrep ⊠ SU2Irrep
    Va = Vect[I]((1, 1, 1//2) => 1)
    Vb = Vect[I]((1, -1, 1//2) => 1)
    Vc = Vect[I]((0, 0, 0) => 1, (0, 0, 1) => 1)
    A = fill_complex!(zeros(ComplexF64, (Va ⊗ Vb) ← Vc), 31)
    perm = () -> begin
        C = permute(A, ((2, 1), (3,)))
        return checksum(C)
    end
    single_ns, single_chk = run_single(perm)
    elapsed = run_bench(perm, config)
    print_result("tensorkit_product_fz2_u1_su2_complex", "single_permute", single_ns, 1, allocated_bytes(perm), single_chk)
    print_result("tensorkit_product_fz2_u1_su2_complex", "warm_permute", elapsed, config.reps, allocated_bytes(perm), perm())
end

config = parse_config()
println("fixture,metric,reps,avg_ns,allocated_bytes,checksum_re,checksum_im")
bench_su2_noncanonical_source(config)
bench_su2_output_permute(config)
bench_product_fz2_u1_su2_complex(config)
