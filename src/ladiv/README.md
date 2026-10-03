# `calaman.ladiv`

LAPACK's `?ladiv`: real-arithmetic complex division. Given real `a, b, c, d`,
compute `p + i*q = (a + i*b) / (c + i*d)` using **Smith's algorithm**, which
divides through by the larger of `|c|`, `|d|` first so the denominator never
overflows when the two parts differ wildly in magnitude — the whole reason
`?ladiv` exists over the naive `(ac + bd) / (c² + d²)`.

It exists to be a **device scalar helper for `laln2`**, which calls it
per-thread. So the arithmetic is header-only and callable from host and device:

```cpp
#include "ladiv/ladiv.h"     // from laln2's own .cu, off the src/ root

T p, q;
calaman::ladiv_scalar<T>(a, b, c, d, &p, &q);   // host or __device__
```

`calaman::ladiv_scalar` is `CLM_HOST_DEVICE` — `__host__ __device__` in a device
pass, empty on a host compile — so the identical body gets device linkage inside
a kernel and compiles for the host for the oracle test. A plain inline template
would not be callable from a `__global__` kernel; that is why it carries the
macro, unlike `calaman.lanv2`'s host-only scalar op. The body uses only
compare / multiply / divide — no library call — so it needs no `<cmath>` and no
`wwr::math`, and reads the same on both backends.

## The batched driver

`?ladiv` is a scalar routine, nothing to offload alone, so the module also
exports an `import`-able driver that batches `n` independent divisions on the
device — the shape the oracle needs to exercise `ladiv_scalar` on a card, and
`calaman.lartg`'s structural sibling:

```cpp
import calaman.ladiv;     // also re-exports calaman::Status
import wwr.runtime_api;

wwr::wwrStream_t stream{};
wwr::wwrStreamCreate(&stream);
// d_a..d_d: length-n device inputs; d_p, d_q: length-n device outputs
calaman::ladiv(stream, n, d_a, d_b, d_c, d_d, d_p, d_q);
```

For each `k` in `[0, n)` it writes `p[k] + i*q[k] = (a[k]+i*b[k]) /
(c[k]+i*d[k])`. Enqueued on `stream` and returns without synchronizing, like a
BLAS call; the caller synchronizes when it needs the outputs. A stream and not a
device handle, because this allocates nothing. The outputs `p`, `q` must be
distinct from each other; an output may alias an input. Every divisor must be
non-zero. `n == 0` is a no-op success.

## Real only

This is the real-arithmetic `?ladiv` (`SLADIV` / `DLADIV`), over `float` and
`double`. The complex-input variants `cladiv` / `zladiv` take two complex
operands and are a different signature, not a trivial instantiation.

## Shape

`ladiv.h` is the header-only scalar kernel (host + device). `interface.cppm` is
the host wrapper for the batched driver (it returns `Status` and re-exports it
from `calaman.error_handling`); `ladiv.cu` is the device half — a single
per-element functor, launched through `wwr.extension.parallel_for`, whose body
just calls `ladiv_scalar`; `ladiv_bridge.h` carries the launcher declaration
across the host/device boundary (a global module fragment cannot `import`).
`instantiations.cpp` explicitly instantiates the driver for each type.

## Tested

`test/ladiv/` is an oracle suite: the device result must agree with reference
LAPACK's `sladiv` / `dladiv` to the shared tolerance. LAPACKE exposes no
`?ladiv` C wrapper (it is a LAPACK auxiliary routine), so the oracle calls the
Fortran symbols `sladiv_` / `dladiv_` directly, reached through
`calaman::lapack_reference` (which links `LAPACK::LAPACK`). The cases cover
`|c| > |d|` and `|c| < |d|` (both Smith branches), purely real and purely
imaginary divisors, mixed signs, and the `n == 0` no-op. It is `REQUIRES_GPU`
(excluded by `ctest -LE gpu`) and builds only when `calaman::lapack_reference`
is present.
