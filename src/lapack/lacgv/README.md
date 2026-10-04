# `calaman.lacgv`

The GPU counterpart of LAPACK's `?lacgv`: conjugate a complex vector in place on
the device. One entry point, templated over the two complex element types:

```cpp
import calaman.lacgv;      // also re-exports calaman::Status
import wwr.runtime_api;
import wwr.complex;

wwr::wwrStream_t stream{};
wwr::wwrStreamCreate(&stream);
// d_x: device vector of n complex elements, stride incx
calaman::lacgv<wwr::wwrDoubleComplex>(stream, n, d_x, 1);
```

Overwrites `x[k*incx]` with `conj(x[k*incx])` for `k` in `[0, n)`. The conjugation
is enqueued on `stream` and returns without synchronizing, like a BLAS call; the
caller synchronizes when it needs `x`. A stream and not a device handle, because
this routine allocates nothing — it needs neither a device index nor a memory
pool, matching `calaman.lacpy`. See `test/shared/README.md` for the full reasoning.

## Complex only

`?lacgv` exists solely as `clacgv` / `zlacgv`: the conjugate of a real vector is
the vector itself, so there is no `s`/`d` variant. The surface is the two complex
types (`wwrFloatComplex`, `wwrDoubleComplex`), constrained by `wwr::complex_fp` —
the same scope `calaman.complex_cast` has. The conjugate goes through
`calaman::device::elem_ops<T>::conj` (`common/elem_ops.cuh`), which spells the
precision-divergent `wwrConj`/`wwrConjf` once and never touches `.x`/`.y` (not
portable to `hipComplex`).

## Stride, and the sign of `incx`

`incx` is the stride between elements; it must be non-zero. Its **sign does not
change the result**: because conjugation is per-element and order-independent, a
negative `incx` conjugates exactly the same elements as its magnitude would. The
device launcher mirrors reference `CLACGV`'s `ioff` stepping — for `incx < 0` it
starts the walk at the far end (`x[(n-1)*|incx|]`) and steps back — so the set of
touched addresses matches LAPACK's for any `incx`.

## Mapping from CLACGV

Changed, per `docs/architecture.md` §4: the `c`/`z` variants become one template
over `ComplexT`. Kept: `N` and `INCX` stay `int` (as `LAPACKE_?lacgv` takes them,
`INCX` signed), since a meaningful negative stride needs the signedness. There is
no `INFO` — `?lacgv` is an auxiliary routine that reports none; this wrapper
returns `calaman::Status` instead, carrying the kernel-launch error if one occurs.

## Shape

`interface.cppm` is the host wrapper (it returns `Status` and re-exports it from
`calaman.error_handling`); `lacgv.cu` is the device half — a single per-element
functor launched through `wwr.extension.parallel_for`; `lacgv_bridge.h` carries
the launcher declaration across the host/device boundary (a global module fragment
cannot `import`). `instantiations.cpp` explicitly instantiates the wrapper for each
type.

## Tested

`test/lacgv/` is an oracle suite: the device result must agree with reference
LAPACK's `LAPACKE_clacgv` / `LAPACKE_zlacgv` computed on the host, over unit and
non-unit strides (including negative), plus the `n <= 0` no-op. It is
`REQUIRES_GPU` (excluded by `ctest -LE gpu`) and builds only when
`calaman::lapack_reference` is present.
