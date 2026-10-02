# calaman.paterson_stockmeyer

Matrix polynomial evaluation in `O(sqrt(d))` matrix products.

Computes the same thing [`calaman.horner`](../horner/README.md) does —

```
p(A) = c_0 I + c_1 A + c_2 A^2 + ... + c_d A^d
```

— with the same ascending device-side coefficient convention, but for a
fraction of the products at moderate degree.

Supported element types: `float`, `double`. (Complex is a deliberate later
extension, for the reason [`src/common/constants.h`](../common/constants.h)
documents: there is no portable `constexpr` spelling of the `kOne`/`kZero` gemm
scalars in complex — the same wall `calaman.horner` stops at.)

## Algorithm

Group the terms in blocks of `s` and read the result as a polynomial in
`X = A^s`:

```
p(A) = sum_{j=0}^{r} B_j X^j        X = A^s,  r = floor(d / s)
B_j  = c_(js) I + c_(js+1) A + ... + c_(js+s-1) A^(s-1)
```

Every `B_j` is a linear combination of the *same* powers of `A`, so those get
built once and shared. The only matrix products are:

```
  (s - 1)          building A^2, A^3, ..., A^s
+ floor(d / s)     Horner steps over X
```

| | Horner | Paterson–Stockmeyer |
|---|---|---|
| `d = 4` | 4 products | **3** |
| `d = 8` | 8 | **4** |
| `d = 16` | 16 | **7** |
| `d = 30` | 30 | **10** |
| workspace | 1 block | `s` blocks |

That last row is the trade. Horner holds one `n x n` scratch block at any
degree; this holds the power bank plus an accumulator.

| Layer | Responsibility |
|---|---|
| `wwr::gemm` (`wwr.wrappers.blas`) | Every matrix product — the power bank and the Horner steps. |
| Custom device launcher (`paterson_stockmeyer.cu`) | Building each `B_j` from the powers in a **single fused pass**. With BLAS this would be `s` separate axpy launches, each a full read-modify-write over `n^2`. Coefficients are read through a device pointer. |

Two details worth relying on:

- **The `B_j` add is free.** Each block is written straight into the gemm's
  *output* buffer, and the product is then accumulated on top with `beta = 1`
  (`P_next = P_cur · X + B_j`). One launch per step, not two, and no temp block.
- **No host synchronization.** Coefficients are read on the device; the gemm
  scalars are the host constants `kOne`/`kZero` from `calaman.common`. A
  coefficient produced by an earlier kernel can be fed straight in.

As in `calaman.horner`, the accumulator ping-pongs between the caller's `P` and
the scratch block, with the starting buffer chosen by the parity of `r` so the
last step lands in `P`. Only the `n x n` block of `P` is written — padding rows
between `n` and `ldp` are untouched.

## Choosing `s`

By default the module picks `s` itself, and `paterson_stockmeyer_plan()` reports
what it chose:

```cpp
struct PatersonStockmeyerPlan {
    int s, r, num_powers, num_blocks, num_gemms;
};
constexpr PatersonStockmeyerPlan paterson_stockmeyer_plan(int degree, int s_requested = 0);
```

The choice is an **exact** argmin of the product count over `s = 1 .. d+1`, not
a rounded `sqrt(d)`. Scanning is free at these degrees and it avoids two traps:

- the floor/ceil edge cases around `sqrt(d)`, and
- the `s > d` regime, where there are no Horner steps at all and the bank only
  has to reach `A^d`. A `sqrt` heuristic never looks there, but it owns the
  bottom of the range: at `d = 2`, `s = 3` forms `A^2` and fuses one block for
  **one** product against Horner's two — and at `d = 1`, `s = 2` evaluates
  `c₀I + c₁A` as a single fused block and multiplies **nothing at all**.

Ties go to the smallest `s`, which is the smaller power bank, so a tie is broken
toward less memory.

Passing a positive `s_requested` overrides the choice — for capping the
workspace when `s` blocks of `n^2` is more memory than you have, or for tuning
against real gemm throughput rather than a product count. It must be passed
identically to the size query and the evaluation. Values above `d + 1` are
clamped; `s = 1` degenerates exactly to Horner.

On product count alone this never loses to Horner at any degree. What
[`calaman.horner`](../horner/README.md) still has is a workspace that does not
depend on the degree — one scratch block, always — and no plan to reason about.
That is the reason to reach for it, not the flop count.

## Pointer mode

`paterson_stockmeyer()` requires the handle's **default (host) pointer mode**,
like `calaman.horner` / `calaman.diff_norm` / `calaman.laqps` / `calaman.geqp3`:
the `gemm` scalars `kOne`/`kZero` are passed by host address. This is
independent of the device-resident coefficients — those are read directly by the
block-build launcher, which takes a device pointer, never through the BLAS
handle.

(The original CUDA-only code saved, overrode and restored the handle's pointer
mode for the duration of the call. That machinery is gone in this port:
WarpWraps's neutral layer exposes no `wwrblasGetPointerMode`, and calaman routes
device-side coefficient reads through its own kernel rather than through the BLAS
pointer mode, so requiring host mode is both portable and consistent with the
rest of the tree. The error paths get simpler for it — there is no mode to
restore on the way out.)

## API

```cpp
import calaman.paterson_stockmeyer;
import wwr.blas;   // wwrblasHandle_t

const std::size_t work_bytes = calaman::paterson_stockmeyer_bufferSize<T>(n, degree);

wwr::wwrblasStatus_t calaman::paterson_stockmeyer<T>(
    wwr::wwrblasHandle_t handle, int n,
    const T* d_coeffs, int degree,   // ascending: c_k * A^k
    const T* d_A, int lda,
    T* d_P, int ldp,                 // out: overwritten
    void* d_work, std::size_t work_bytes,
    int s_requested = 0);            // 0 = choose automatically
```

`d_coeffs` holds `degree + 1` entries in **ascending** order: `d_coeffs[k]` is
the coefficient of `A^k`, and `d_coeffs[0]` contributes `c_0 I`.

`d_work` must be 256-byte aligned, as device allocations and memory-pool
allocations are. `d_A`, `d_P` and `d_work` must not overlap.

Bad dimensions, a null pointer or an undersized workspace return
`WWRBLAS_STATUS_NOT_INITIALIZED` — the only neutral non-success code WarpWraps
exposes (see `calaman.diff_norm`). Otherwise the return is the first failing
`gemm` status, or `WWRBLAS_STATUS_SUCCESS`.

### Example

```cpp
// p(A) = sum_k c_k A^k, degree 12.
const std::size_t work_bytes = calaman::paterson_stockmeyer_bufferSize<double>(n, 12);
// d_work: work_bytes of 256-aligned device scratch

calaman::paterson_stockmeyer<double>(handle, n, d_c, 12,
                                     d_A, n, d_P, n,
                                     d_work, work_bytes);

// What did that cost? 6 products and 3 blocks, against Horner's 12 and 1.
constexpr auto plan = calaman::paterson_stockmeyer_plan(12);
static_assert(plan.s == 3 && plan.num_gemms == 6 && plan.num_blocks == 3);
```

## Limitations

- **Workspace grows with the degree.** `s` blocks of `n^2`, where
  `calaman.horner` needs one. At large `n` this is the binding constraint, and
  capping `s` gives back products one for one.
- **The products are still sequential.** The power bank chains
  (`A^k = A^(k-1) · A`) and the Horner steps chain, so nothing overlaps. A
  binary-powering bank would shorten the dependency chain at equal product
  count, which would matter for small `n` where launches dominate.
- **No degree-aware conditioning work.** Like `calaman.horner`, this evaluates
  what it is given; if `p` is ill-conditioned at `A`, blocking does not help. It
  is slightly *less* accurate than Horner in general, since the explicit powers
  `A^k` are formed directly rather than reached one multiplication at a time.
- **Not the true optimum.** For a general polynomial the minimum number of
  non-scalar products is `sqrt(2d)`-ish (Paterson–Stockmeyer's own bound with
  preprocessing), which needs coefficient-dependent preprocessing this does not
  do.
- **`float`/`double` only** — complex is a deliberate later extension.

## Build

`CMakeLists.txt` builds two targets: `calaman.paterson_stockmeyer.device`
(static, `paterson_stockmeyer.cu`, links `wwr.extension.parallel_for`) and the
module library `calaman.paterson_stockmeyer`. The device archive is pulled in
whole (`$<LINK_LIBRARY:WHOLE_ARCHIVE,...>`) so the explicit
`device::paterson_stockmeyer_build_block` instantiations survive. Template
instantiations are declared `extern template` in `interface.cppm`, defined once
in `instantiations.cpp` (host) and `paterson_stockmeyer.cu` (device).

**Dependencies:** `wwr.wrappers.blas`, `wwr.blas`, `wwr.runtime_api`,
`calaman.common`, and `wwr.extension.parallel_for` (device side).

## Tests

`test/test/paterson_stockmeyer/paterson_stockmeyer_tests.cpp`, run by
`paterson_stockmeyer_tests`. The oracle is an independent host accumulation of
the power series via reference CBLAS gemm — a different algorithm from the
blocked scheme, so agreement is real evidence — guarded on
`calaman::lapack_reference`.

Host-only suites (no card, run under `ctest -LE gpu`):

- **Plan** — the README cost table (`d = 4, 8, 16, 30`; the `d = 12` example);
  the `d = 0, 1, 2` edges; the chosen `s` is a true argmin over all `s` in
  `[1, d+1]` for every degree up to 32; and `s_requested` override, clamping and
  the `s = 1` Horner degeneration.
- **BufferSize** — the workspace is exactly `plan.num_blocks` aligned blocks, and
  degree 0 / 1 need none.
- **ArgCheck** — every bad argument (including `s_requested < 0`) returns
  `WWRBLAS_STATUS_NOT_INITIALIZED` before any device work.

Device suites (`REQUIRES_GPU`, labeled `gpu`): agreement with the oracle at
degree 0/1/2 and at even/odd outer-Horner parity; a larger `n`; the **all-`s`
sweep** for one polynomial (every block size from the degenerate Horner end to
the fully-explicit end must agree); a direct cross-check against
`calaman.horner`; the zero polynomial (exact); a nilpotent matrix (the series
terminates); and leading-dimension padding left intact.
