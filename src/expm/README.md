# calaman.expm

Matrix exponential `exp(A)` by **scaling and squaring** with a **diagonal Padé
approximant**, choosing the degree from the matrix 1-norm. Four fused device
kernels evaluate the Padé polynomial and build the 1-norm; the wrapped BLAS
(`wwr::gemm`/`geam`) supplies the matrix products and the wrapped LU
solve (`wwr::getrf`/`getrs`) the linear solve. Written once against WarpWraps's
backend-neutral `wwr*` names and built for either vendor; templated over all four
element types (`float`, `double`, and the two complex types), constrained by
`wwr::usual_fp`.

## Algorithm

Higham's Algorithm 2.3 (Higham, *The Scaling and Squaring Method for the Matrix
Exponential Revisited*, SIAM J. Matrix Anal. Appl. 26(4), 2005):

1. Take the **cheapest degree** `m ∈ {3, 5, 7, 9, 13}` whose backward-error
   threshold `θ_m` already covers `norm_1(A)`.
2. Only if even `θ_13` does not, **scale**: pick the smallest `s` with
   `norm_1(A) / 2^s ≤ θ_13`, evaluate `X = r_13(A / 2^s)`, and square `s` times.

With `p_m(x) = Σ b_j x^j` and `q_m(x) = p_m(-x)`, the approximant
`r_m(x) = q_m(x)^-1 p_m(x)` matches `exp(x)` through order `x^(2m)`, with

```
b_j = (2m-j)! * m! / ( (2m)! * j! * (m-j)! )
```

Splitting `p_m` into even and odd halves,

```
V = b0*I + b2*A2 + b4*A4 + ...
U = A * (b1*I + b3*A2 + b5*A4 + ...)
```

gives `p_m(A) = V + U` and `q_m(A) = V - U`, so a single LU solve of
`(V - U) X = (V + U)` produces `r_m(A)`.

**The sign flip is the whole trick.** Because `q = p(-A)`, the denominator costs
nothing beyond the elementwise pass that already builds the numerator. A general
polynomial evaluator — [`calaman.horner`](../horner/README.md) — cannot see that
relationship and would have to evaluate `q` from scratch, which is why it is not
used here despite being cheaper on a *single* polynomial.

### Degree 13

Degree 13 uses Higham's nested form rather than the plain split:

```
U = A[ A6(b13*A6 + b11*A4 + b9*A2) + b7*A6 + b5*A4 + b3*A2 + b1*I ]
V =    A6(b12*A6 + b10*A4 + b8*A2) + b6*A6 + b4*A4 + b2*A2 + b0*I
```

which reaches degree 13 from a power bank that stops at `A^6`. Each parity is
**one fused pass** — writing the inner combination and its matching tail
together — followed by one gemm that multiplies by `A6` and accumulates the tail
with `beta = 1`, so the tail add is free. Running the two parities in sequence
lets them share a single scratch block, which holds degree 13 to the **same six
`n*n` blocks that degree 9 needs**.

### Cost

`pade_num_gemms(m) + s` matrix products:

| `m` | products | `θ_m` (double) | `θ_m` (single) |
|-----|----------|----------------|----------------|
| 3   | 2 | `1.495585217958292e-2` | `4.258730034897931e-1` |
| 5   | 3 | `2.539398330063232e-1` | `1.880152698533769` |
| 7   | 4 | `9.504178996162932e-1` | `3.925724846433284` |
| 9   | 5 | `2.097847961257067`    | `6.249156334514102` |
| 13  | 6 | `5.371920351148152`    | `1.124873763647540e1` |

`expm_plan<T>()` reports the degree, the squarings and the product count for a
given norm before the call.

### θ_m

`θ_m` is the largest `x` satisfying Higham's backward-error criterion

```
sum_{k=2m+1}^{inf} |h_k| x^(k-1) <= u,     h(x) = log(exp(-x) r_m(x))
```

Meeting it means `r_m` applied to the scaled matrix is the exact exponential of
a nearby matrix `A + E` with `norm(E) <= u * norm(A)`. The values come from
evaluating that series in exact rational arithmetic (`h = -x + log p_m - log
q_m`, the logs recovered from `L'f = f'`) and bisecting for the root at 60
significant digits, truncating the tail at `k = 200`. At `u = 2^-53` this
reproduces Higham's published table exactly at `m = 7, 9, 13`, which is the
check that validates the single-precision column.

### Balancing (not built in)

`expm` does **not** balance the matrix itself. Balancing is the orthogonal
similarity `exp(A) = D exp(D^-1 A D) D^-1`, which composes *around* `expm` rather
than belonging inside it. A caller who wants it does the transform explicitly:

1. Run [`calaman.gebal`](../gebal/README.md) with `GebalJob::Scale` to form
   `B = D^-1 A D` in place and recover the diagonal of `D`. Only the **scaling**
   half is wanted (the permutation half isolates eigenvalues, which does nothing
   for the norm); `gebal` scales only by powers of two, so the similarity is
   exact in floating point.
2. Call `expm` on `B`.
3. Re-wrap the result: `exp(A) = D · exp(B) · D^-1`, two diagonal scalings
   (`wwr::dgmm`, with `D` and `D^-1` widened to the element type).

Every factor of two balancing takes off the 1-norm is **one squaring — one
`n^3` product — removed** from the tail, and it improves accuracy on badly
scaled matrices. Skip it on a well-scaled matrix (it cannot help), or when
Watkins' caveat applies (*A case where balancing is harmful*, ETNA 2006).

Keeping balancing out of `expm` leaves the routine a pure `exp(A)`: no hidden
`gebal` dependency, no "keep only if the norm fell" heuristic, two fewer stream
synchronizations, and a caller who does not need it pays nothing.

## No gemm3m

The reference this was ported from offered `cublasGemm3m` (Gauss's
three-multiplication complex product, 25% fewer flops) behind a `use_gemm3m`
option. That is a **cuBLAS-only** call with no hipBLAS counterpart — it lives in
`wwr.cuda.cublas_v2`, not the backend-neutral layer — so routing through it
would compile on exactly one backend and break the promise this project is built
on. The option is gone; every matrix product goes through the portable
`wwr::gemm`.

## API

| Function | Description |
|----------|-------------|
| `expm_bufferSize<T>(cusolver, n, &bytes)` | Device workspace for `expm` (the ladder's worst case) |
| `expm<T>(cublas, cusolver, stream, n, d_A, lda, d_expA, lde, d_work, bytes, d_info, plan)` | `exp(A)` by scaling and squaring |
| `expm_plan<T>(norm1)` | `{m, s, num_gemms}` for a given 1-norm, without evaluating anything |
| `pade_theta<T>(m)` | Backward-error threshold for degree `m` |
| `pade_bufferSize<T>(cusolver, m, n, &bytes)` | Device workspace for `pade` at degree `m` |
| `pade<T>(cublas, cusolver, stream, m, n, d_A, lda, d_r, ldr, d_work, bytes, d_info)` | Unscaled `r_m(A)`; never synchronizes |
| `matrix_norm1<T>(stream, n, d_A, lda, d_colsum)` | Maximum absolute column sum, returned on the host |

`T` is `float`, `double`, `wwr::wwrFloatComplex` or `wwr::wwrDoubleComplex`. All
matrices are column-major. `d_info` is a device array of **2** ints: `[0]`
receives the `getrf` info code, `[1]` the `getrs` one.

Every function returns `calaman::Status`, following `calaman.geqp3`: the one
cross-domain return type carries each WarpWraps result in its OWN domain, so a
solver failure (`getrf` / `getrs`) now surfaces as a solver-domain Status and a
`gebal` runtime failure as a runtime one, instead of the old `INTERNAL_ERROR`
masquerade through the BLAS enum. A bad argument or an undersized workspace is
still reported with an explicit BLAS-domain `INVALID_VALUE` / `ALLOC_FAILED`.

```cpp
struct ExpmPlan { int m; int s; int num_gemms; };   // both the prediction and the report
```

One struct, two roles: `expm_plan<T>(norm1)` returns the `ExpmPlan` a given norm
*will* cost, and `expm()` fills an `ExpmPlan *` with the one it *did* execute.
They are the same type because `expm` is deterministic — the degree and the
squarings follow from the 1-norm alone, so a separate "info" struct could only
restate a subset of this one. There is no options struct: `expm` takes no tuning
knobs now that balancing lives outside it.

Both handles must be bound to `stream`, in the handle's default (host) pointer
mode. The BLAS pointer mode is saved on entry and restored on exit, so callers
using device-pointer scalars are unaffected.

### Workspace

Seven `n*n` blocks: one for the scaled matrix, plus the six that degree 13
needs, plus the `O(n)` column-sum vector. The first block does double duty — it
holds the scaled matrix, then the ping-pong target for the squaring phase —
because `pade()` has finished reading it before the squaring begins.

That aliasing is also why the squaring phase does **not** start by parity to
avoid a final copy the way [`calaman.horner`](../horner/README.md) does: the
scratch is still the *input* at the moment `pade()` would have to write into it.
One block of `n^2` is worth more than one `geam`.

`pade_bufferSize` scales with the degree — three blocks at `m = 3`, six at `m =
9` and `m = 13` — so a caller who knows its matrices are small-normed can ask
for far less than `expm_bufferSize` reserves.

### Synchronization

`expm` **synchronizes the stream once**, to read `norm_1(A)` onto the host: the
degree and the scaling exponent drive host-side control flow. `pade` never
synchronizes; call it directly when the norm is known in advance and a fully
asynchronous path is required.

## Custom kernels

`expm.cu` holds the four genuinely per-element pieces, built on
`wwr.extension.parallel_for` and `calaman.reduce_columns`:

| Launcher | Purpose |
|--------|---------|
| `pade_even_odd` | Reads up to four powers once and writes **two** real-coefficient combinations of them. Every degree on the ladder is built from this one launcher: for `m ≤ 9` it produces `V` and `W` in a single call, and for `m = 13` it is called twice, each pairing one nested inner combination with its tail. The block count `NP` is a template parameter so the accumulation unrolls. |
| `pade_split` | Reads `U` and `V` once, writing `V + U` to the output (arbitrary `ldr`) and `V - U` over `V`. The output may alias `V`: element `i` is read by the one thread that writes it. |
| `abs_colsums` | Per-column absolute sums for the 1-norm, via `calaman.reduce_columns` with the **true modulus** as the pre-transform (not the `|Re| + |Im|` surrogate). |
| `max_reduce` | Single-block max reduction over those column sums, in place, NaN-propagating. |

**No Thrust.** The max reduction and the column sums are a hand-written kernel
and `calaman.reduce_columns` respectively, following the decision WarpWraps's
`parallel_for.cuh` and `calaman.reduce_columns` both document: a Thrust
*algorithm* broke under relocatable device code, so this codebase reduces by
hand. (The original of this routine hit exactly that failure — a
`thrust::reduce` with `thrust::maximum` raised
`cudaErrorInvalidDeviceFunction` across the full device link — and replaced it
with the hand-written kernel this port keeps.)

## Files

| File | Role |
|------|------|
| `interface.cppm` | Module interface: the ladder, workspace layouts, `expm`, `pade`, `expm_plan`, `matrix_norm1` |
| `expm_bridge.h` | Padé coefficient tables and the kernel-launcher declarations |
| `expm.cu` | The five custom kernels |
| `instantiations.cpp` | Explicit instantiations for the four element types |

## Tests

`test/expm/expm_tests.cpp`. There is no `LAPACKE_?expm` — LAPACK ships no
matrix exponential — so the oracle is closed forms, identities that hold for any
matrix, and an independent approximation from a different code path:

- **closed forms** — `exp(0) = I`, `exp(diag) = diag(exp)`, and a nilpotent `N`
  whose series terminates.
- **`exp(A) exp(-A) = I`** and **`exp(A) = exp(A/2)^2`**, identities a correct
  exponential satisfies regardless of the matrix.
- **the degree ladder** — for a norm inside each rung, the result is compared
  against `r_13` evaluated on the same matrix (an independent coefficient table
  and code path), so a wrong coefficient anywhere on the ladder shows up.
- **the plan, padded leading dimensions, pointer-mode restoration and argument
  validation.**

The numerical suites stage the matrices on the device and run the kernels, so
they are `REQUIRES_GPU` (labeled `gpu`, excluded by `ctest -LE gpu`); the plan,
threshold and argument-checking cases are host-only.
