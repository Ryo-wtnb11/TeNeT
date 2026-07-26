using Pkg; Pkg.activate(@__DIR__)
using CategoryData, TensorKitSectors, Printf

const objs = collect(values(Object{Fib}))
c(z) = (real(complex(z)), imag(complex(z)))
p(io, tag, keys, z) = @printf(io, "%s %s %s %s\n", tag, join(keys, " "), repr(c(z)[1]), repr(c(z)[2]))

open("fixture.txt", "w") do io
    println(io, "# CategoryData.jl Fib = PMFC{2,1,0,2,0,0}; 1-based object ids as in the source files")
    println(io, "# pkg CategoryData v0.3.6 rev e793a08a093f6ba890a32c7e57d8e8b347441058")
    println(io, "# TensorKitSectors v", Pkg.dependencies()[Base.UUID("13a9c161-d5da-41f0-bcbd-e1a08ae0647f")].version)
    println(io, "# artifact data-v0.1.3 sha256 586f48411e0c3aa8780959d2395724bf6d1de4e4bcf785eb84864e0d5c045cf6")
    println(io, "# unit ", unit(Object{Fib}).id)
    for a in objs
        @printf(io, "dual %d %d\n", a.id, dual(a).id)
        p(io, "dim", (a.id,), dim(a))
        p(io, "frobeniusschur", (a.id,), frobeniusschur(a))
        p(io, "twist", (a.id,), twist(a))
    end
    for a in objs, b in objs, c_ in objs
        @printf(io, "N %d %d %d %d\n", a.id, b.id, c_.id, Nsymbol(a, b, c_))
        p(io, "R", (a.id, b.id, c_.id), Rsymbol(a, b, c_))
        p(io, "A", (a.id, b.id, c_.id), Asymbol(a, b, c_))
        p(io, "B", (a.id, b.id, c_.id), Bsymbol(a, b, c_))
    end
    for a in objs, b in objs, c_ in objs, d in objs, e in objs, f in objs
        p(io, "F", (a.id, b.id, c_.id, d.id, e.id, f.id), Fsymbol(a, b, c_, d, e, f))
    end
end
println("ok")
