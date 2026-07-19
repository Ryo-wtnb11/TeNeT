# Exact SU(2) authority

TeNeT delegates SU(2) Wigner 6j evaluation and Regge canonicalization to
`wigner-symbols 0.5.1`. TeNeT retains the tensor-network convention boundary:
checked doubled-spin labels, TensorKit-compatible argument order, phase and
dimension factors, bounded publication of the final `f64`, fusion-rule identity,
and in-process cache invalidation.

The supported public range is `0 <= 2j <= 254`. The dependency's canonical key
stores components as `u8`, and its complete-domain sizing reserves the following
value. TeNeT rejects a fusion whose closure would exceed this range instead of
silently defining a truncated SU(2) category.

## Build and license audit

- `wigner-symbols 0.5.1` is licensed MIT or Apache-2.0 and does not declare an
  MSRV.
- The current resolution, `rug 1.30.0`, declares Rust 1.85 and LGPL-3.0+.
- `gmp-mpfr-sys 1.7.1` declares Rust 1.71 and LGPL-3.0+. It builds native GMP
  code, so clean builds require a C toolchain and are materially slower than
  ordinary Rust-only builds.
- All three authority crates are exact-pinned because this library repository
  intentionally does not track `Cargo.lock`. The canonical key is
  version-sensitive, and even a one-ULP conversion change would invalidate
  cached coefficient bits. Updating any pin therefore requires incrementing
  `SU2_EXACT_AUTHORITY_VERSION`, which participates in the in-process rule and
  operation-cache identity.
- Binary distributors must satisfy the applicable LGPL obligations for the
  linked GMP/Rug components, including notices and the required replacement or
  relinking mechanism. This audit records the dependency boundary; it is not a
  substitute for a release-specific license review.

CI sets `GMP_MPFR_SYS_CACHE` and caches that directory by OS, the resolved Rust
toolchain fingerprint, and Cargo manifests/lockfile. The cache contains compiled
native dependency artifacts only; it is not an SU(2) coefficient cache.
