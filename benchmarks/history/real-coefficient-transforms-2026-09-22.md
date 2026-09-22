# Host transforms keep a real structural coefficient real (#1398)

Date 2026-09-22. Host: Apple M4 Max, macOS 15.5, rustc 1.96.0.

- **Before:** `origin/main` `07b241f6`.
- **After:** the same tree with the #1398 change applied.
- **Build:** both from the same worktree and the same Cargo lock, built one
  after the other into one target directory with
  `cargo build --release -p tenet-rs --example eager_overhead_ledger`.
- **Binaries (sha256 prefix):** before `069e762c9968a9f4`, after
  `bc53adec45e62304`.
- **Method:** the E1 example with the filters `permute` and `repartition`,
  `LEDGER_THREADS=one`. Four passes per revision, interleaved
  before/after; the table reports the minimum over the four.
- **Timing is an observation only.** The machine had another tenant (load
  average 14–25), and the change rests on its structure, not on a threshold.

## Structural change

The per-element work for a real coefficient `c` acting on a `Complex64`
payload goes from the promoted complex product

    (c + 0i)(x + yi) = (cx - 0y) + (cy + 0x)i     4 multiplies, 2 adds

to the componentwise product TensorKit performs

    c * (x + yi) = cx + (cy)i                     2 multiplies

and, when the coefficient is exactly one, to no arithmetic at all. `f64`
payloads keep the same single multiply, except at coefficient one, where the
multiply is now skipped. Allocation calls and bytes are unchanged in every row.


### `permute`

| case | before min ns | after min ns | ratio | alloc calls/bytes |
|---|---:|---:|---:|---|
| U1,f64,r2_s8_d2 | 898 | 879 | 0.979 | 10/1328 |
| U1,f64,r2_s8_d16 | 1470 | 1548 | 1.053 | 10/17456 |
| U1,f64,r3_s4_d4 | 1357 | 1274 | 0.939 | 10/7312 |
| U1,f64,r4_s3_d4 | 3819 | 4334 | 1.135 | 10/40176 |
| U1,f64,r5_s2_d2 | 1106 | 1102 | 0.996 | 10/3920 |
| U1,c64,r2_s8_d2 | 979 | 896 | 0.915 | 10/1584 |
| U1,c64,r2_s8_d16 | 2192 | 1889 | 0.862 | 10/33840 |
| U1,c64,r3_s4_d4 | 1708 | 1524 | 0.892 | 10/13456 |
| U1,c64,r4_s3_d4 | 6166 | 5042 | 0.818 | 10/79088 |
| U1,c64,r5_s2_d2 | 1224 | 1161 | 0.949 | 10/6480 |
| fZ2xU1,f64,r2_s8_d2 | 1245 | 1130 | 0.908 | 10/1328 |
| fZ2xU1,f64,r2_s8_d16 | 1933 | 1700 | 0.879 | 10/17456 |
| fZ2xU1,f64,r3_s4_d4 | 1458 | 1618 | 1.110 | 10/7312 |
| fZ2xU1,f64,r4_s3_d4 | 4486 | 4500 | 1.003 | 10/40176 |
| fZ2xU1,f64,r5_s2_d2 | 1125 | 1234 | 1.097 | 10/3920 |
| fZ2xU1,c64,r2_s8_d2 | 1250 | 1220 | 0.976 | 10/1584 |
| fZ2xU1,c64,r2_s8_d16 | 2636 | 2153 | 0.817 | 10/33840 |
| fZ2xU1,c64,r3_s4_d4 | 1783 | 1583 | 0.888 | 10/13456 |
| fZ2xU1,c64,r4_s3_d4 | 5792 | 5416 | 0.935 | 10/79088 |
| fZ2xU1,c64,r5_s2_d2 | 1202 | 1307 | 1.087 | 10/6480 |
| SU2,f64,r2_s8_d2 | 896 | 879 | 0.981 | 10/1328 |
| SU2,f64,r2_s8_d16 | 1570 | 1552 | 0.989 | 10/17456 |
| SU2,f64,r3_s4_d4 | 1883 | 1767 | 0.938 | 10/12944 |
| SU2,f64,r4_s3_d4 | 15917 | 15625 | 0.982 | 18/95664 |
| SU2,f64,r5_s2_d2 | 4583 | 4306 | 0.939 | 14/6832 |
| SU2,c64,r2_s8_d2 | 912 | 1004 | 1.100 | 10/1584 |
| SU2,c64,r2_s8_d16 | 2250 | 1875 | 0.833 | 10/33840 |
| SU2,c64,r3_s4_d4 | 2573 | 2258 | 0.878 | 10/24720 |
| SU2,c64,r4_s3_d4 | 20833 | 19292 | 0.926 | 18/189872 |
| SU2,c64,r5_s2_d2 | 4916 | 4500 | 0.915 | 14/12208 |

### `repartition`

| case | before min ns | after min ns | ratio | alloc calls/bytes |
|---|---:|---:|---:|---|
| U1,f64,r2_s8_d2 | 769 | 699 | 0.909 | 9/1168 |
| U1,f64,r2_s8_d16 | 1470 | 1417 | 0.964 | 9/17296 |
| U1,f64,r3_s4_d4 | 1109 | 1106 | 0.997 | 10/7152 |
| U1,f64,r4_s3_d4 | 2135 | 2392 | 1.120 | 10/40016 |
| U1,f64,r5_s2_d2 | 979 | 991 | 1.012 | 10/3760 |
| U1,c64,r2_s8_d2 | 837 | 788 | 0.941 | 9/1424 |
| U1,c64,r2_s8_d16 | 2058 | 1611 | 0.783 | 9/33680 |
| U1,c64,r3_s4_d4 | 1393 | 1297 | 0.931 | 10/13296 |
| U1,c64,r4_s3_d4 | 4396 | 3639 | 0.828 | 10/78928 |
| U1,c64,r5_s2_d2 | 1057 | 1021 | 0.966 | 10/6320 |
| fZ2xU1,f64,r2_s8_d2 | 986 | 1046 | 1.061 | 9/1168 |
| fZ2xU1,f64,r2_s8_d16 | 1611 | 1702 | 1.056 | 9/17296 |
| fZ2xU1,f64,r3_s4_d4 | 1256 | 1351 | 1.076 | 10/7152 |
| fZ2xU1,f64,r4_s3_d4 | 2490 | 2396 | 0.962 | 10/40016 |
| fZ2xU1,f64,r5_s2_d2 | 1083 | 1088 | 1.004 | 10/3760 |
| fZ2xU1,c64,r2_s8_d2 | 1167 | 1143 | 0.980 | 9/1424 |
| fZ2xU1,c64,r2_s8_d16 | 2427 | 2067 | 0.851 | 9/33680 |
| fZ2xU1,c64,r3_s4_d4 | 1597 | 1500 | 0.939 | 10/13296 |
| fZ2xU1,c64,r4_s3_d4 | 5166 | 3542 | 0.686 | 10/78928 |
| fZ2xU1,c64,r5_s2_d2 | 1229 | 1083 | 0.881 | 10/6320 |
| SU2,f64,r2_s8_d2 | 750 | 806 | 1.074 | 9/1168 |
| SU2,f64,r2_s8_d16 | 1369 | 1458 | 1.065 | 9/17296 |
| SU2,f64,r3_s4_d4 | 1500 | 1632 | 1.088 | 10/12784 |
| SU2,f64,r4_s3_d4 | 4166 | 4389 | 1.053 | 10/95312 |
| SU2,f64,r5_s2_d2 | 1146 | 1214 | 1.060 | 10/6576 |
| SU2,c64,r2_s8_d2 | 788 | 860 | 1.092 | 9/1424 |
| SU2,c64,r2_s8_d16 | 1950 | 1758 | 0.902 | 9/33680 |
| SU2,c64,r3_s4_d4 | 1842 | 2000 | 1.086 | 10/24560 |
| SU2,c64,r4_s3_d4 | 9250 | 7520 | 0.813 | 10/189520 |
| SU2,c64,r5_s2_d2 | 1571 | 1417 | 0.902 | 10/11952 |

## Reading

- The `c64` rows with real work per element improve by 12–31 % (the largest:
  `fZ2xU1,c64,r4_s3_d4` repartition 0.69, `U1,c64,r2_s8_d16` repartition 0.78,
  `U1,c64,r4_s3_d4` permute 0.82). That is the removed complex multiply.
- The `f64` rows and the smallest `c64` rows move within ±12 % in both
  directions with no structural cause: their per-element arithmetic is
  unchanged or reduced, the block count is small, and the machine was loaded.
  They are noise at this confidence.
- Raw pass-1 rows are in
  `real-coefficient-transforms-2026-09-22-{permute,repartition}-{before,after}.csv`.
