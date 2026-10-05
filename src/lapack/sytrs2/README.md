# calaman.sytrs2 — symmetric Bunch-Kaufman solve, level-3

`calaman.sytrs2` is LAPACK's `?sytrs2`: the same solve as `calaman.sytrs`
(`A X = B` for a Bunch-Kaufman-factored symmetric `A`), but **level-3**. Instead
of the per-column `ger`/`gemv` walk, it batches the pivot interchanges to the ends
so the two triangular solves become single `trsm` calls over all right-hand sides.

**Complex is symmetric, not Hermitian** — no conjugation. The Hermitian cousin is
`calaman.hetrs2`.

## Why it needs `?syconv`

The factor `?sytrf` stores mixes two things in one subdiagonal entry: the
unit-triangular `L` and, for a 2×2 pivot, `D`'s off-diagonal. A `trsm` with
`DIAG=UNIT` would wrongly treat that `D` off-diagonal as an `L` entry. And the
stored `L` is in the *interleaved-interchange* order the level-2 solve walks, not
the order a single batched `trsm` needs.

`?syconv` fixes both, in place, and the solve reverts it before returning:

- **CONVERT** — **VALUE** (a device kernel): move each 2×2 block's `D`
  off-diagonal out of the factor into the `n`-element `E` workspace, zeroing it in
  the factor so the triangle is pure unit-`L`. **PERMUTATIONS** (host-driven
  `wwr::swap`): reorder the factor's off-diagonal triangle into batched-`trsm`
  order.
- **REVERT** — the exact inverse (permutations then value), so `A` is byte-for-byte
  the factor again on return.

## The solve

Between convert and revert, mirroring reference `?sytrs2`:

```
P^T B  →  trsm(L, uplo, N, Unit)  →  D^-1 B  →  trsm(L, uplo, T, Unit)  →  P B
```

`P^T B` / `P B` are the pivot interchanges applied to `B` (host-driven `swap`).
`D^-1 B` is the block-diagonal apply — the 1×1 reciprocal scale and the symmetric
2×2 solve — with the 2×2 off-diagonal now read from `E`. These are the two device
kernels in `sytrs2.cu` (structurally `calaman.sytrs`'s, kept separate on the
module boundary, as `pstrf` keeps `pstf2`'s).

## Workspace

One `n`-element device buffer (`d_work`, the `E` vector) — the only cost over the
level-2 path's zero extra workspace. `sytrs2_bufferSize<T>(n)` returns `n`.

## The oracle

`LAPACKE_?sytrf` factors a symmetric indefinite matrix; the device `sytrs2` and
`LAPACKE_?sytrs2` then solve from the same factor, and the two `X` agree to
tolerance (and `A X` reproduces `B`). The suite also checks `A` is unchanged after
the convert/revert round-trip.
