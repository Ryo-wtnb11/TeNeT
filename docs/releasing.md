# Releasing TeNeT

This is the checklist for a future registry release. The facade is
`tenet-rs` 0.1.1; the workspace pins the Tenferro 0.7.1 line. Its Rust library
target remains `tenet`, so
downstream code imports from `tenet::typed` and `tenet::sector`.

## Publication order

Publish only after the previous layer is visible in the crates.io index. Each
package is packaged and inspected before it is published.

1. `tenet-sectors` (after a compatible Racah release is available)
2. `tenet-core`
3. `tenet-dense`
4. `tenet-operations`
5. `tenet-tensors`
6. `tenet-matrixalgebra`
7. `tenet-rs`
8. `tenet-network`
9. `tenet-category-data`

## Checks for each package

Run from a clean checkout with the tracked `Cargo.lock` (CI also uses `--locked`) and the intended toolchain:

```sh
cargo package -p <package> --locked
cargo publish --dry-run -p <package> --locked
```

Inspect the packaged manifest and reject the package if a normal dependency
still points to a local path or a mutable git branch. Path dependencies with a
matching version are allowed during workspace development; the registry
manifest must resolve them from crates.io.

After the complete closure is published, build a clean downstream fixture:

```toml
[dependencies]
tenet = { package = "tenet-rs", version = "0.1.1" }
```

The fixture must compile `use tenet::typed::{Runtime, TensorMap}` without a sibling checkout,
Tenferro source override, or Racah git override.

## External prerequisites

The workspace manifests pin the `tenferro-*` crates at 0.7.1 (`tenet-dense/Cargo.toml`),
`t4a-cubecl-runtime` at `=0.10.1` (kept in lockstep with Tenferro), and `racah` at 0.2.4
(`tenet-sectors/Cargo.toml`).
The registry-only gate must still verify compatible resolved versions before a
future release.
