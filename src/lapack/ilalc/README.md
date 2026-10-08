# calaman.ilalc

The GPU counterpart of LAPACK's `ila?lc` (`ilaslc`/`iladlc`/`ilaclc`/`ilazlc`):
the last non-zero column of an `m`-by-`n` column-major matrix. Its row twin is
[`calaman.ilalr`](../ilalr/README.md), and the two share one contract.

## The result is a count

The Fortran returns a 1-based column index. Here the same integer is read as
**the number of leading columns that must be considered** — `1 +` the 0-based
column of the last non-zero — so a caller trimming a reflector application
(`?larf`'s `LASTC`) uses it directly as a length. An all-zero matrix gives `0`
under both readings. An empty matrix (`m` or `n` zero) also gives `0`; the
reference reads `A(1,N)` when `M = 0`, so that one case is defined here rather
than copied.

"Non-zero" is the reference's `A(i,j).NE.ZERO`: a NaN counts, `-0` does not, and
a complex entry counts when either component is non-zero. Rows `m..lda-1` (the
padding between columns) are never read.

## Shape

Identical to `calaman.ilalr`: a `calaman::Status`-returning host wrapper
(`interface.cppm`) over a device translation unit (`ilalc.cu`) through a
launcher bridge (`ilalc_bridge.h`), with the per-column intermediate (`n` ints)
in a caller-provided workspace sized by `ilalc_bufferSize(m, n)`.

The scan is a two-stage max-reduction: one block per column writes `j + 1` when
column `j` holds a non-zero (`0` otherwise), then `calaman.reduce_columns` folds
the `n` values with `max`. The result is written to a device `int` and the call
does not synchronize. The reference's corner fast path is not reproduced.
