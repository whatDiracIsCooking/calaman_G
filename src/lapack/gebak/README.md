# `calaman.gebak`

The GPU counterpart of LAPACK's `?gebak` and the back-transformation companion of
[`calaman.gebal`](../gebal/README.md): given the eigenvectors of a balanced matrix
`B = D^-1 P^T A P D`, recover the eigenvectors of the original `A` by undoing the
diagonal scaling `D` and the symmetric permutation `P`, in place, on the `n x m`
column-major matrix `V`. Templated over all four element types — `float`,
`double`, `wwrFloatComplex`, `wwrDoubleComplex`.

```cpp
import calaman.gebak;
import wwr.runtime_api;      // wwrStream_t, wwrStreamCreate, wwrStreamSynchronize

wwr::wwrStream_t stream{};
wwr::wwrStreamCreate(&stream);

// d_V: n x m device matrix (ldv); d_scale, ilo, ihi: calaman::gebal's outputs,
// fed through unchanged.
calaman::gebak(stream, calaman::GebakJob::Both, calaman::GebakSide::Right, n,
               ilo, ihi, d_scale, m, d_V, ldv);
wwr::wwrStreamSynchronize(stream);   // gebak does NOT synchronize on its own
```

`GebakJob` selects which halves to undo — `None` (`'N'`), `Permute` (`'P'`),
`Scale` (`'S'`), `Both` (`'B'`) — and must match the `job` passed to `gebal`.
`GebakSide` says whether `V` holds `Right` (`'R'`) or `Left` (`'L'`) eigenvectors:
right vectors scale row `i` by `scale(i)`, left vectors by `1/scale(i)`. `ilo`,
`ihi` are 1-based and `d_scale` is **real even for complex `T`**
(`wwr::ComplexToRealType<T>`), carrying `gebal`'s dual encoding unchanged — the
diagonal of `D` inside `[ilo, ihi]`, the 1-based interchange index outside it. It
returns a `calaman::Status`.

**`gebak` is asynchronous.** It only enqueues its kernels on `stream` and returns
without synchronizing — the mirror image of `gebal`, which must synchronize
because it branches on device data. The caller synchronizes before reading `d_V`.

## Why complex is a first-class path here

For the same reason as `gebal`: the back-transform only multiplies by the **real**
scales `D` and moves rows — there is no complex `tau` or conjugation to differ, as
there is for a reflector (`larfg`, `laqp2`). So the four-type surface is genuine,
and it rides WarpWraps's neutral complex (`wwr.complex` on the host, `complex.h` in
the kernel) with no backend leak.

## Mapping from `?gebak`

Kept: the name, `ilo`/`ihi` and their 1-based convention, the dual `scale`
encoding, and — unlike `gebal` — the **exact numerics**. There is no
balancing-convention divergence here: `gebak` is a verbatim scaling-and-swapping
of `V`, so it agrees with a current reference LAPACK bit for bit. Changed, per
`docs/architecture.md` §4: `CHARACTER JOB`/`SIDE` become the typed `GebakJob` /
`GebakSide` enums, the `s/d/c/z` variants become one template over `T`, and `INFO`
becomes a returned `calaman::Status`. A single-index window (`ilo == ihi`) carries
no scaling, matching LAPACK's `GO TO 30`.

## Shape

`interface.cppm` is the module: the `GebakJob` / `GebakSide` enums and the **host
driver** — thin control flow that validates, then issues the two kernels. Unlike
`gebal`'s driver it reads **nothing** back from the device. `gebak.cu` is the
device half: the two kernels and one launcher each. `gebak_bridge.h` carries the
launcher declarations across the host/device boundary (a global module fragment
cannot `import`); to keep it parseable in both the host GMF and the device `.cu` —
which have no common complex header — every launcher is generic over `<T>` (and
the real type `<R>`), never naming a complex type. `instantiations.cpp` explicitly
instantiates the driver for each type.

**Algorithm shape.** *Scaling* is full-grid parallel: rows `ilo..ihi` are
multiplied by `scale(i)` (or `1/scale(i)` for left vectors) over all columns.
These rows sit inside the balanced window and the permutation touches only rows
outside it, so the two stages are disjoint. *Permutation* runs in **one
cooperating block**: `?gebak`'s interchange sequence is both data-dependent (each
target `k = int(scale(i))` is read from the array) and order-dependent
(consecutive swaps can share a row), so it cannot parallelise across the grid. One
block replays the exact `ii = 1..n` walk — the low block `[1, ilo)` in reverse,
the high block `(ihi, n]` forward, self-maps skipped — with a `__syncthreads()`
between swaps, cooperating on the `O(m)` column work of each. Keeping the replay on
the device is what lets the driver stay asynchronous.

## Tested

`test/gebak/gebak_tests.cpp` (`GebakOracleTests`, `REQUIRES_GPU`), run for all
four element types. Because `gebak` has no convention divergence, the suite is an
**oracle diff** against `LAPACKE_?gebak` (not a spec test like `gebal`), so it
guards on `calaman::lapack_reference` and skips when that target is absent. All
scale factors are powers of two, so every operation is exact and the comparison is
**bit for bit**.

- **Scaling** — both sides over a multi-row window (`V * scale(i)` and
  `V * 1/scale(i)`), with `ldv > n`.
- **Permutation** — the subtle index walk: an `n = 6, ilo = 3, ihi = 4` case
  drives swaps in both the reverse-walked low block and the forward high block,
  with a self-map skipped.
- **Both** and the **`ilo == ihi`** window where LAPACK skips scaling.
- **Quick returns** — `JOB = None` and `m = 0`.
