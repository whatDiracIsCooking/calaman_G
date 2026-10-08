# calaman.ilalr

The GPU counterpart of LAPACK's `ila?lr` (`ilaslr`/`iladlr`/`ilaclr`/`ilazlr`):
the last non-zero row of an `m`-by-`n` column-major matrix. Its column twin is
[`calaman.ilalc`](../ilalc/README.md).

## The result is a count

The Fortran returns a 1-based row index. Here the same integer is read as **the
number of leading rows that must be considered** — `1 +` the 0-based row of the
last non-zero — so there is no off-by-one to translate: a caller trimming a
reflector (`?larf`'s `LASTV`) uses it directly as a length. An all-zero matrix
gives `0` under both readings. An empty matrix (`m` or `n` zero) also gives `0`;
the reference reads `A(M,1)` when `N = 0`, so that one case is defined here
rather than copied.

"Non-zero" is the reference's `A(i,j).NE.ZERO`: a NaN counts, `-0` does not, and
a complex entry counts when either component is non-zero. Rows `m..lda-1` (the
padding between columns) are never read.

## Shape

A `calaman::Status`-returning host wrapper (`interface.cppm`) over a device
translation unit (`ilalr.cu`) through a launcher bridge (`ilalr_bridge.h`), with
an explicit-instantiation unit, like `calaman.lange`. Unlike `lange` it
allocates nothing: the per-column intermediate (`n` ints) lives in a
caller-provided workspace sized by `ilalr_bufferSize(m, n)` and carved through
`carve_workspace` (the `workspace` skill).

The scan is a two-stage max-reduction: one block per column writes the count for
that column, then `calaman.reduce_columns` folds the `n` counts with `max`. The
result is written to a device `int` and the call does not synchronize. The
reference's fast path — checking the two corners of the last row before
scanning backwards — buys nothing on a GPU and is not reproduced.
