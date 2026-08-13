# tenet-sectors

Fusion-rule providers and sector types: U(1), Z/N, fermion parity, SU(2),
Fibonacci, products, and feature-gated SUN providers. Providers own sector
labels and categorical coefficients; tensor algorithms consume their traits.

`ProductFusionRule<L, R>` is itself a provider: `left.product(right)` creates
an ordered, recursively nestable product, and `ProductSector` is its typed
sector label. Products do not flatten or reorder factors automatically.

Provider capability traits determine which typed operations are available.
In particular, complex categorical coefficients (for example Fibonacci) are
distinct from a tensor payload's scalar type and are supported only on the
Host paths whose bounds admit them.
