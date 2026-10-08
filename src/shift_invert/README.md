# calaman.shift_invert

`DenseShiftInvert<T>` — a [`linear_operator`](../linear_operator/linear_operator.cppm)
model of `(A - sigma I)^{-1}` for a dense real symmetric `A`, the operator a
shift-invert eigensolver runs on to reach the eigenvalues nearest `sigma`
(`lambda = sigma + 1/theta`).

Its own module rather than a lanczos partition, so any `linear_operator`
consumer (lanczos, davidson) can take it, and so an inexact (Krylov) model can
sit beside it later ([`docs/architecture.md` §9](../../docs/architecture.md)).

| Step | What runs |
|---|---|
| `prepare(stream)` | copy `A`'s `uplo` triangle (`calaman.lacpy`), subtract `sigma` on the diagonal (`calaman.laset` + `axpy`), Bunch-Kaufman factor once (`wwr::sytrf`); a singular shift is a failing `Status` |
| `apply(stream, k, X, Y)` | `Y = X`, then solve against the factor (`calaman.sytrs2`) |

Real `float`/`double` only. The workspace follows the carve-once convention:
`dense_shift_invert_bufferSize` sizes it, `make_dense_shift_invert_slices`
carves it into `DenseShiftInvertSlices`.
