# `calaman.gebal`

The GPU counterpart of LAPACK's `?gebal`: balance a general `n x n` column-major
matrix for the nonsymmetric eigenproblem. Overwrites `A` with
`B = D^-1 P^T A P D`, isolating eigenvalues that already sit on the diagonal and
equilibrating what is left so a subsequent Hessenberg reduction and QR iteration
see far smaller eigenvalue condition numbers. Templated over all four element
types — `float`, `double`, `wwrFloatComplex`, `wwrDoubleComplex`.

```cpp
import calaman.gebal;
import wwr.runtime_api;      // wwrStream_t, wwrStreamCreate

wwr::wwrStream_t stream{};
wwr::wwrStreamCreate(&stream);

int lwork = 0;
calaman::gebal_bufferSize<double>(n, &lwork);   // ints of device scratch: 2n + 1
// d_A: n x n device matrix, lda; d_scale: length n; d_work: length lwork ints
int ilo = 0, ihi = 0;
calaman::gebal(stream, calaman::GebalJob::Both, n, d_A, lda, &ilo, &ihi,
               d_scale, d_work);
```

`GebalJob` selects which halves to run — `None` (`'N'`), `Permute` (`'P'`),
`Scale` (`'S'`), `Both` (`'B'`). `ilo` and `ihi` are host outputs and 1-based, as
in LAPACK. `d_scale` is **real even for complex `T`** (`wwr::ComplexToRealType<T>`)
and carries LAPACK's two overloaded conventions unchanged, so the array feeds
`?gebak` as-is: inside `[ilo, ihi]` it holds the diagonal of `D`; outside it holds
the 1-based index that position was exchanged with.

**`gebal` synchronizes `stream`.** Both stages branch on device data — which row
is isolated next, whether a sweep changed anything — so the routine is
host-driven and returns with its outputs complete and nothing left queued.
`GebalJob::None` is the one path that stays asynchronous. It returns
`wwr::wwrError_t` (`wwrSuccess`, or the first runtime error).

## Why complex is a first-class path here

Every other `calaman` routine is `float`/`double` only and defers complex,
because for a reflector (`larfg`, `laqp2`) the complex case differs *materially*
— a complex `tau`, a conjugated row. Balancing has no such difference: it only
measures magnitudes and applies **real** power-of-two scales. Complex magnitudes
use `CABS1` (`|Re| + |Im|`), matching `CGEBAL`/`ZGEBAL`. So the four-type surface
is genuine, not a trivial instantiation, and it rides WarpWraps's neutral complex
(`wwr.complex` on the host, `complex.cuh` in the kernel) with no backend leak.

## Mapping from `?gebal`

Kept: the name, `ilo`/`ihi` and their 1-based convention, and the dual `scale`
encoding (both are facts of the `?gebak` contract, not Fortran accommodations).
Changed, per `docs/architecture.md` §4: `CHARACTER*1 JOB` becomes the typed
`GebalJob` enum, the `s/d/c/z` variants become one template over `T`, and `INFO`
becomes a returned `wwr::wwrError_t`.

### Differences from a modern reference LAPACK

- Magnitudes use the **off-diagonal 1-norm** over the active window, as in
  EISPACK `BALANC` and LAPACK through 3.4. LAPACK 3.5+ switched to a diagonal-
  inclusive 2-norm, so scale factors can differ from a current reference by a
  radix step.
- An index whose norms are not finite is **skipped** rather than reported: a NaN
  makes every comparison in the search loops false, which would otherwise hang
  the kernel. Newer LAPACK returns `INFO = -3` for that case.
- `max_sweeps` caps the loop. The sweep is a strict descent so LAPACK runs it
  uncapped; the cap only prevents a pathological input from spinning the host
  loop. The default is `calaman::gebal_default_max_sweeps`.

Balancing is not unconditionally beneficial — see Watkins, "A case where
balancing is harmful" (ETNA, 2006). `GebalJob::Permute` alone is the conservative
choice when that matters.

## Shape

`interface.cppm` is the module: the `GebalJob` enum, `gebal_bufferSize`, and the
**host driver** — the control flow plus the `wwr.runtime_api` memcpy/memset/sync
that reads the device back between launches. `gebal.cu` is the device half: every
kernel (mark / pick / swap / sweep, plus the `set_int`, `record_perm` and
`parallel_for` fill-ones helpers) and one launcher per kernel. `gebal_bridge.h`
carries the launcher declarations across the host/device boundary (a global
module fragment cannot `import`); to keep it parseable in both the host GMF and
the device `.cu` — which have no common complex header — every launcher is
generic over `<T>` (and the real type `<R>`), never naming a complex type.
`instantiations.cpp` explicitly instantiates the driver for each type.

**Algorithm shape.** *Permutation* is full-grid parallel: one `mark_nonzero` pass
flags every row/column with an off-diagonal non-zero, a reduction picks the next
row/column that isolates an eigenvalue (largest index for rows, smallest for
columns, matching LAPACK's scan), and two swap kernels apply the symmetric
transposition over full rows/columns (the out-of-range entries are provably zero,
so this matches LAPACK's sub-range swap). The swaps run full-length where LAPACK
swaps only a sub-range; the extra entries are provably zero, so the two agree.

*Scaling* is the Parlett-Reinsch iteration in **one cooperating block**. The
sweep is Gauss-Seidel — index `i` is scaled before `i+1` measures its norms — and
that dependency is why it cannot parallelise across the grid: a simultaneous
(Jacobi) update does not converge (on `[[1, 2^10], [2^-10, 1]]` the imbalance
just oscillates). The block cooperates on the `O(n)` work per index, so a sweep
is `O(n^2)` on one SM — fine at the moderate `n` where balancing is worth doing.
Scale factors are exact powers of two, so the similarity is exact in floating
point: undoing it reproduces the input bit for bit.

## Tested

`test/test/gebal/gebal_tests.cpp` (`GebalSpecTests`, `REQUIRES_GPU`), run for all
four element types. The suite is **spec-based**, not an oracle diff: because the
1-norm convention can differ from a current reference LAPACK by a radix step, it
asserts the specification instead and needs no reference LAPACK.

- **Exact reproduce** — undoing the reported permutation and scaling reconstructs
  the input *bit for bit* (the scale factors are powers of two, so the similarity
  is exact). The host replays gebal's own swap order and the diagonal `D`. This is
  the headline test; it covers permutation and scaling together, dense (no
  isolation) and with a real transposition.
- **Analytic 2×2** — `[[1, 2^10], [2^-10, 1]]` balances to `[[1, 1], [1, 1]]` with
  `D = diag(2^10, 1)`, the one closed-form case.
- **Idempotence (stationarity)** — a second `Scale` pass over a balanced matrix
  accepts nothing (`B` unchanged, every factor `1`); the negative control is the
  2×2, whose first pass demonstrably *did* change it.
- **Structural** — a diagonal matrix isolates every eigenvalue (`ilo == ihi == 1`),
  and `n = 1` is the trivial fixed point.
