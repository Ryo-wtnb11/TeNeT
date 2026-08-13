# Releasing TeNeT

This is the planned procedure for TeNeT's first registry release; it is not
evidence that the release has been published. The procedure uses the published
Tenferro 0.3.0 line. The
facade is published as `tenet-rs`; its Rust library target remains `tenet`, so
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

`tenet-krylov` is not part of the initial public closure and remains private
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
tenet = { package = "tenet-rs", version = "0.1.0" }
```

The fixture must compile `use tenet::prelude::*` without a sibling checkout,
Tenferro source override, or Racah git override.

## External prerequisites

Tenferro 0.3.0 is the selected initial release line and is already published.
Racah must provide a compatible registry release before `tenet-sectors` can
pass the registry-only gate. Tenferro 0.4 is a future upgrade, not a blocker
for this release procedure.
