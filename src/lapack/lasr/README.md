# calaman.lasr

The GPU counterpart of LAPACK's `?lasr`: apply the `k-1` plane rotations
`(c[j], s[j])` — exactly what `calaman.lartg` produces — to the `m`-by-`n`
column-major matrix `A`, for every combination of

| selector | values | meaning |
|---|---|---|
| `Side` | `L` / `R` | `A <- P A` (`k = m`) / `A <- A P^T` (`k = n`) |
| `Pivot` | `V` / `T` / `B` | rotation `j` acts in plane `(j, j+1)` / `(1, j+1)` / `(j, k)` |
| `Direct` | `F` / `B` | `P = P(k-1) ... P(1)` / `P = P(1) ... P(k-1)` |

Templated over `float`, `double`, `wwrFloatComplex` and `wwrDoubleComplex`. As
in `ZLASR`, the rotations stay real for a complex `A`: `c` and `s` are `T`'s
real component type (`ComplexToRealType<T>`), so `calaman.lartg`'s output feeds
either family.

## Two entry points

- **`calaman::lasr<T>(stream, side, pivot, direct, m, n, c, s, A, lda)`** — the
  module (`import calaman.lasr;`, link `calaman::lasr`). One launch: the lines
  of the unrotated dimension are split into slabs, one block per slab, and each
  block runs `lasr_block`.
- **`calaman::lasr_block<T, R>(side, pivot, direct, m, n, c, s, A, lda)`** —
  `lasr.h`, a `__device__` helper for a kernel that applies the rotations from
  inside its own launch (`?steqr` updating `Z`). Every thread of the block must
  call it; it opens and closes with `__syncthreads()`. A `.cu` `#include`s
  `"lapack/lasr/lasr.h"` and its device library links `calaman::lasr::header`.
  `lasr_line` (one thread, one line) is in the same header.

## Parallel shape

The rotation chain is sequential along the rotated dimension, but each line of
the other dimension (a column for `Side::L`, a row for `Side::R`) is touched only
by its own chain. So a thread carries whole lines through every rotation, and no
barrier is needed between rotations — only around the call, so inputs and
results are visible block-wide.
