# `calaman.lasrt`

The GPU counterpart of LAPACK's `?lasrt`: sort a 1-D array on the device, in
place, into increasing or decreasing order. One entry point, templated over
`float` and `double`:

```cpp
import calaman.lasrt;   // also re-exports calaman::SortDir
import wwr.runtime_api;

wwr::wwrStream_t stream{};
wwr::wwrStreamCreate(&stream);
// d_d: device array of n elements
calaman::lasrt(stream, calaman::SortDir::I, n, d_d);   // increasing
```

`SortDir` selects the direction — `I` (increasing, `d(0) <= ... <= d(n-1)`) or
`D` (decreasing, `d(0) >= ... >= d(n-1)`). The sort is enqueued on `stream` and
returns without synchronizing, like a BLAS call and like `calaman.laset`; the
caller synchronizes when it needs `d`. `n <= 1` is already sorted: the wrapper
enqueues nothing, succeeds, and leaves `d` untouched (it may be null). `n < 0`
returns an invalid-value `Status`.

`SortDir` lives in `calaman.common` (its `:enums` partition, shared with the
device `.cu` through `common/enums.h`) and is re-exported by `calaman.lasrt`, so
`import calaman.lasrt;` alone names it — the same arrangement `calaman.laset`
uses for `Region`.

A stream and not a device handle, because this routine allocates nothing: it
needs neither a device index nor a memory pool, matching `calaman.laset` /
`calaman.lacpy`. See `test/shared/README.md` for the full reasoning.

## Mapping from DLASRT

Kept: the name and the length `n` (`int` — the sort's explicit stack is sized in
`n`, as LAPACK's is). Changed, per `docs/architecture.md` §4: the
`CHARACTER*1 ID` becomes the typed `calaman::SortDir` enum (`common/enums.h`),
and the `s/d` variants become one template over `T`.

There is no `INFO` out-parameter. DLASRT reports two illegal-argument codes;
both fall away here. `INFO = -1` (a bad `ID`) is unreachable with a typed
`SortDir`, and `INFO = -2` (a negative `N`) becomes an invalid-value `Status`
return — the same trade the typed selectors make across `src/`.

Real only (`float`/`double`): a sort needs a total order, which the complex
types do not have, so LAPACK ships no `c`/`z` `?lasrt`. Unlike `calaman.laset`'s
complex extension, there is no type to add here.

## Shape

`interface.cppm` is the host wrapper (it re-exports `SortDir` from
`calaman.common`); `lasrt.cu` is the device half; `lasrt_bridge.h` carries the
launcher declaration across the host/device boundary (a global module fragment
cannot `import`) and passes `SortDir` across it by `#include`. `instantiations.cpp`
explicitly instantiates the wrapper for each type.

The sort is one hand-launched **single-thread** kernel. The algorithm is
inherently sequential — quicksort with an explicit 32-frame stack (LAPACK's
`STACK(2,32)`), reverting to insertion sort on ranges of 20 or fewer, with a
median-of-3 pivot and a Hoare partition — and it branches on the data it is
reordering at every step, so one thread runs the whole sort rather than fanning
out. This is LAPACK's `dlasrt` ported verbatim to 0-based inclusive ranges;
pushing the larger sub-range first keeps the stack depth logarithmic, so 32
frames admit every `int n`.

The sort body itself is `lasrt_serial<T, Dir>(d, n)` in `lasrt.cuh`, a
`__device__` function the kernel merely calls — so another kernel can run the
same sort in-thread (`?sterf` sorts its eigenvalues this way, and only when its
iteration converged). A downstream `.cu` includes `"lapack/lasrt/lasrt.cuh"` and
its device library links `calaman::lasrt::header` (the `src/` root).

A sorted array is a function of the input **multiset** and the direction alone,
so this agrees with the reference `?lasrt` bit for bit regardless of which
comparison sort either side runs — which is what lets the test compare with `==`
rather than a tolerance. A parallel bitonic or merge sort is a possible later
optimisation; it would change the running time, not the result.

## Tested

`test/lasrt/` — an oracle suite against the reference `LAPACKE_?lasrt`, bit for
bit (`==`), over both directions and a spread of sizes that straddle the
insertion/quicksort threshold, plus duplicates, already-sorted and
reverse-sorted inputs, and the `n <= 1` early return. `REQUIRES_GPU` (labeled
`gpu`, excluded by `ctest -LE gpu`) and guarded on `calaman::lapack_reference` at
configure time, so a missing oracle is a missing tier, not a silent pass.
