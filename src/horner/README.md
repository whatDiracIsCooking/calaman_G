# calaman.horner

Matrix polynomial evaluation by Horner's method.

Computes

```
p(A) = c_0 I + c_1 A + c_2 A^2 + ... + c_d A^d
```

for a square column-major matrix `A`, with the coefficients held on the device.

Supported element types: `float`, `double`. (Complex is a deliberate later
extension, for the reason `src/common/constants.h` documents: there is no
portable `constexpr` spelling of the `kOne`/`kZero` gemm scalars in complex.)

## Algorithm

The Horner recurrence, nested from the leading coefficient inward:

```
P <- c_d I
for k = d-1, d-2, ..., 0:
    P <- P A + c_k I
```

which unfolds to `p(A) = (...((c_d A + c_{d-1}) A + c_{d-2}) A + ...) A + c_0`.

Each step is one `gemm` plus a diagonal update, so a degree-`d` polynomial costs
**`d` matrix products** — the minimum for a general dense evaluation, and half
the traffic of the naive form (which needs `d` products *and* `d` axpy passes
over `n^2` elements, with every power of `A` live at once).

| Layer | Responsibility |
|---|---|
| `wwr::gemm` (`wwr.wrappers.blas`) | the `gemm` at each step — all of the arithmetic |
| Custom device launchers (`horner.cu`) | `P <- c_d I` to seed the recurrence, and `P(i,i) += c_k` to close each step. Both read the coefficient from a **device** pointer. |

Two properties fall out of the recurrence and are worth relying on:

- **One `n x n` scratch block, whatever the degree.** Each step consumes the
  previous accumulator, so nothing but the running `P` has to stay live.
- **No host synchronization.** The coefficients are read on the device by the
  diagonal launchers, and the `gemm` scalars are the host constants `kOne`/`kZero`
  from `calaman.common`. A coefficient produced by an earlier kernel can be fed
  straight in.

A `gemm` cannot write one of its own inputs, so the accumulator ping-pongs
between the caller's `P` and the scratch block. The starting buffer is chosen by
the parity of the degree so that the final step lands in `P` — an odd-degree
evaluation starts in the workspace. This is why there is no final copy.

Only the `n x n` block of `P` is written; the padding rows between `n` and `ldp`
are left untouched, so `P` may be a view into a larger allocation.

## Pointer mode

`horner()` requires the handle's **default (host) pointer mode**, like
`calaman.diff_norm` / `calaman.laqps` / `calaman.geqp3`: the `gemm` scalars
`kOne`/`kZero` are passed by host address. This is independent of the
device-resident coefficients — those are read directly by the diagonal
launchers, which take a device pointer, never through the BLAS handle.

(The original CUDA-only code saved, overrode and restored the handle's pointer
mode so it could read the coefficients device-side via cuBLAS. That machinery is
gone in this port: WarpWraps's neutral layer exposes no `wwrblasGetPointerMode`,
and calaman routes device-side coefficient reads through its own kernels rather
than through the BLAS pointer mode, so requiring host mode is both portable and
consistent with the rest of the tree.)

## API

```cpp
import calaman.horner;
import wwr.blas;   // wwrblasHandle_t

const std::size_t work_bytes = calaman::horner_bufferSize<T>(n);   // one n*n block

calaman::Status calaman::horner<T>(
    wwr::wwrblasHandle_t handle, int n,
    const T* d_coeffs, int degree,  // ascending: c_k * A^k
    const T* d_A, int lda,
    T* d_P, int ldp,                // out: overwritten
    void* d_work, std::size_t work_bytes);
```

`d_coeffs` holds `degree + 1` entries in **ascending** order: `d_coeffs[k]` is
the coefficient of `A^k`, and `d_coeffs[0]` contributes `c_0 I`.

`d_work` must be 256-byte aligned, as device allocations and memory-pool
allocations are. A degree-0 polynomial is just `c_0 I` — no product, so a null
buffer is accepted in that case.

`d_A`, `d_P` and `d_work` must not overlap.

Bad dimensions, a null pointer or an undersized workspace return
`WWRBLAS_STATUS_NOT_INITIALIZED` — the only neutral non-success code WarpWraps
exposes (see `calaman.diff_norm`). Otherwise the return is the first failing
`gemm` status, or `WWRBLAS_STATUS_SUCCESS`.

## Limitations

- **Horner is not asymptotically optimal in the degree.**
  Paterson–Stockmeyer evaluates a degree-`d` polynomial in `O(sqrt(d))` products
  by grouping terms, and wins for roughly `d > 8` — at the cost of keeping
  `sqrt(d)` powers of `A` resident. Horner is the right choice for the low
  degrees that show up in Padé numerators and rational filters, and whenever
  extra `n x n` workspace is the binding constraint.
- **The `d` products are strictly sequential.** Nothing in the recurrence can
  overlap, so for small `n` the evaluation is launch-latency bound.
- **No Horner-with-shift** (evaluation at `A - sigma I`) and no synthetic
  division; this computes the value only.
- **`float`/`double` only** — complex is a deliberate later extension.

## Build

`CMakeLists.txt` builds two targets: `calaman.horner.device` (static,
`horner.cu`, links `wwr.extension.parallel_for`) and the module library
`calaman.horner`. The device archive is pulled in whole
(`$<LINK_LIBRARY:WHOLE_ARCHIVE,...>`) so the explicit `device::horner_*`
instantiations survive. Template instantiations are declared `extern template`
in `interface.cppm`, defined once in `instantiations.cpp` (host) and `horner.cu`
(device).

**Dependencies:** `wwr.wrappers.blas`, `wwr.blas`, `wwr.runtime_api`,
`calaman.common`, and `wwr.extension.parallel_for` (device side).

## Tests

Not yet wired. A suite under `test/` would, per element type, check: the
workspace query; the degree-0 and degree-1 closed forms; agreement with an
explicitly-accumulated power series at even and odd degree (the two ping-pong
parities); exactness on a nilpotent matrix, where the series terminates; the
zero polynomial; leading-dimension padding left intact; and the
argument-checking contract — against the reference LAPACK as the oracle, like
the other numerical suites.
