# calaman.diis

Pulay DIIS (direct inversion of the iterative subspace) extrapolation for a
fixed-point iteration, over a device-resident history. Written for an SCF loop's
Fock matrix, but the vectors are opaque: any length-`len` iterate with a residual
that vanishes at the fixed point works.

## Module

`calaman.diis`

## Design

DIIS replaces the newest iterate `F` with `F* = sum_i c_i F_i` over the recent
history, the coefficients minimizing the norm of `sum_i c_i e_i` subject to
`sum_i c_i = 1`. They solve the bordered system

```
[ <e_i|e_j>  -1 ] [ c ]   [ 0  ]
[   -1^T      0 ] [ l ] = [ -1 ]
```

**State and workspace.** The ring cursor (`len`, `cap`, `size`, `head`) is a
host `DiisState`; everything else lives in one caller-owned device workspace,
carved once (`DiisSlices`, sized by `diis_bufferSize`): the `(len x cap)` vector
and residual histories (column-major, `lda = len`), the persistent `cap x cap`
Gram (leading dimension `cap`), and the `cap` coefficients. The same workspace
must back the same state on every push.

**Circular storage.** The newest pair overwrites the oldest physical column and
`head` advances, so nothing is shifted on eviction. Eviction begins only once the
ring is full, so the valid columns are always the prefix `[0, size)` (growth,
`head == 0`) or the whole buffer — never a wrapped range — which keeps every BLAS
operand a single base pointer with a fixed stride. Columns are in physical
(rotated) order, not age order.

**Incremental Gram.** Only the inner products with the just-written column
`slot` change, so each push recomputes that row and column with two `gemv`
(`trans = T`) writing straight into the upper triangle the solve reads
(`gram[min*cap + max]`):

- `j = 0..slot` → column segment and diagonal, `y = gram + slot`, `incy = cap`;
- `j = slot..size-1` → row segment, `y = gram + slot*cap + slot`, `incy = 1`.

This runs on every push, including the first, so every triangle entry `[a][c]`
is rewritten by the push that wrote column `max(a, c)`. Nothing is read stale,
and `diis_reset` (cursor only) is safe because regrowth refills every entry
before it is read.

**The solve.** With `size >= 2` the `(size+1)`-square bordered system is solved
by a one-block kernel (`diis.cu`): it builds the system in shared memory and
thread 0 runs partial-pivot Gaussian elimination. The Gram block is first divided
by its largest diagonal entry, which puts it on the Lagrange border's unit scale
(better conditioned, and `c` is unchanged — only the multiplier rescales), so the
singularity test is relative: a pivot below `pivot_tol` (default
`kDiisPivotTol<T>` = `1024 eps`), or an all-zero Gram, marks the subspace
singular. An absolute floor would miss a duplicated residual whose rounding
residue scales with `|B|`, and would falsely flag every push once residuals are
small late in convergence. On a singular subspace the kernel emits the unit
selector at the newest physical column, so the extrapolation reproduces the
plain newest `F`. The caller's optional `singular_device` flag records which
case happened, on the device. `cap` is bounded by `kDiisMaxHistory` (64) so the
system fits the default shared-memory budget.

**Extrapolation.** `F* = F_hist c` is one `gemv` over physical columns
`0..size-1`. The sum is order-invariant, so the rotated column order is fine.

The routine never synchronizes and reads nothing back to the host. The BLAS
handle is forced to host pointer mode for the call (`ScopedPointerMode`); the
coefficients are a `gemv` data operand, so they never leave the device.

## Why a custom kernel, not `calaman.sysv`

The bordered matrix is symmetric indefinite, so `calaman.sysv` could factor it.
But the system is tiny (`cap + 1`), the fallback has to *select* a vector on the
device when a pivot is merely small (`sytrf` flags only an exact zero), and a
vendor factorization would add a workspace query and several launches per push.
One block solving in shared memory is both cheaper and gives the fallback a
place to live.

## Port notes

Ported from a cuBLAS-specific SCF helper class that owned its device buffers and
borrowed the handle per call. To fit calaman:

- the class became free functions over a host `DiisState` and a caller-owned,
  carve-once workspace (the shipped surface allocates nothing);
- the raw `cublasDgemv`/CUDA calls became `wwr::gemv`/`wwr*` runtime calls, and
  the kernel launches `WWR_WARP_SIZE` threads with `__syncthreads`, so it builds
  for either backend; it is templated over `float`/`double`;
- the internal host readback of the singular flag was dropped — the flag is
  handed back on the device instead (the `info_device` convention of
  `calaman.inv_sqrt`), keeping the push fully stream-ordered;
- the history cap is validated against `kDiisMaxHistory` rather than allowed to
  overrun shared memory.

## Files

| File | Role |
|------|------|
| `interface.cppm` | Module interface — `DiisState`, `diis_bufferSize`, `diis_push_and_extrapolate` |
| `instantiations.cpp` | Explicit `float`/`double` instantiations |
| `diis.cu` | The bordered-solve kernel |
| `diis_bridge.h` | GMF-shared `diis_solve` launcher declaration |
| `CMakeLists.txt` | Build configuration |
