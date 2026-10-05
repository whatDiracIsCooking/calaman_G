# calaman.hetrs2 — Hermitian Bunch-Kaufman solve, level-3

`calaman.hetrs2` is LAPACK's `?hetrs2`: the same solve as `calaman.hetrs`
(`A X = B` for a Bunch-Kaufman-factored Hermitian `A`), but **level-3** — the pivot
interchanges are batched to the ends so the two triangular solves become single
`trsm` calls. **Complex only** (`c`/`z`); a real Hermitian matrix is symmetric
(`calaman.sytrs2`).

## Mostly `sytrs2`, with the Hermitian twists

The reference `?hetrs2` reuses the **symmetric** `?syconv` (the VALUE move is
unconjugated — the off-diagonal stored in `E` is the raw `D` entry; the conjugation
lives elsewhere). So the `?syconv` convert/revert (the VALUE kernel plus the
host-driven PERMUTATION swaps on the factor's triangle) and the `P^T B` / `P B`
pivot loops are **identical** to `calaman.sytrs2`. `hetrs2` diverges only in:

- **the second `trsm`** — conjugate-transpose (`OP_C`), realizing `U^H`/`L^H`
  (`sytrs2` uses `OP_T`);
- **the `D^-1` apply** — Hermitian: the 1×1 divides by the **real** diagonal, and
  the 2×2 conjugates the off-diagonal on one row (the reference's `DCONJG(AKM1K)`),
  reading that off-diagonal from `E`.

```
?syconv(C)  →  P^T B  →  trsm(uplo, N, Unit)  →  D^-1 B  →  trsm(uplo, C, Unit)  →  P B  →  ?syconv(R)
```

Those device stages (the `?syconv` VALUE move and the two `D^-1` applies) live in
`hetrs2.cu`, structurally `calaman.hetrs`'s, kept separate on the module boundary.

## Workspace

One `n`-element device buffer (`d_work`, the `E` vector). `hetrs2_bufferSize<T>(n)`
returns `n`.

## The oracle

Exercised through `calaman.hesv`, which uses `hetrs2` when the workspace allows:
`LAPACKE_?hesv` on the same Hermitian indefinite system, checked by the backward-
error residual. `A` is left unchanged by the convert/revert round-trip.
