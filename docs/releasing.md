# Releasing TeNeT

This is the checklist for a future registry release. The published facade is
`tenet-rs` 0.1.1 and uses the published Tenferro 0.3.0 line. Its Rust library
target remains `tenet`, so
downstream code continues to write `use tenet::prelude::*`.

## Publication order

Publish only after the previous layer is visible in the crates.io index. Each
package is packaged and inspected before it is published.

1. `tenet-sectors` (after a compatible Racah release is available)
2. `tenet-core`
3. `tenet-dense`
4. `tenet-operations`
5. `tenet-tensors`
6. `tenet-matrixalgebra`
7. `tenet-macros`
8. `tenet-rs`
9. `tenet-network`
10. `tenet-category-data`

`tenet-krylov` is outside this publication closure and remains unpublished
until the facade uses it.

## Checks for each package

Run from a clean checkout with the intended lockfile and toolchain:

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

The fixture must compile `use tenet::prelude::*` without a sibling checkout,
Tenferro source override, or Racah git override.

## External prerequisites

Tenferro 0.3.0 and Racah 0.1.1 are the current published dependency lines.
The registry-only gate must still verify compatible resolved versions before a
future release.
