# `calaman.laln2`

The GPU counterpart of LAPACK's `?laln2`: solve the small, scaled linear system

```
(ca * op(A) - w * D) X = SCALE * B
```

for the `NA`-by-`NA` matrix `X`, with `NA` either `1` or `2`. `A` is `NA`-by-`NA`
real, `ca` is a real scalar, `D = diag(d1, d2)` is real diagonal, and
`w = wr + i*wi` is **real** (`nw == 1`) or **complex** (`nw == 2`). `op(A)` is `A`
or `Aᵀ` (selected by `ltrans`). When `w` is complex, `X` and `B` are `NA`-by-`2`
slabs — column 1 the real part, column 2 the imaginary part. `SCALE` (`<= 1`)
guards the solution against overflow. One entry point, templated over the two
real element types:

```cpp
import calaman.laln2;      // also re-exports calaman::Status
import wwr.runtime_api;

wwr::wwrStream_t stream{};
wwr::wwrStreamCreate(&stream);
// d_a, d_b: device inputs; d_x, d_scale, d_xnorm, d_info: device outputs
calaman::laln2<double>(stream, /*ltrans=*/false, /*na=*/2, /*nw=*/1,
                       /*smin=*/smin, /*ca=*/1.0, d_a, 2, /*d1=*/d1, /*d2=*/d2,
                       d_b, 2, /*wr=*/wr, /*wi=*/0.0, d_x, 2,
                       d_scale, d_xnorm, d_info);
```

The solve is enqueued on `stream` and returns without synchronizing, like a BLAS
call; the caller synchronizes when it needs the outputs. A stream and not a
device handle, because this routine allocates nothing — it needs neither a device
index nor a memory pool, matching `calaman.lasy2`.

## Everything is on the device

`?laln2` is called deep inside `?trevc` / `?laqtr` for one eigenvector at a time,
where its inputs already live on the device. So the whole algorithm — the real
and complex `1x1` cases, the real and complex `2x2` cases, the complete-pivoting
Gaussian elimination, the `SMINI` perturbation of a near-singular `C`, and the
`SCALE` / `XNORM` computation — runs in a **single device thread**; there is no
host round-trip. That is why the four outputs (`x`, `scale`, `xnorm`, `info`) are
device pointers, not host scalars: nothing is ever read back here.

## Real element type, w may be complex

The reference is `SLALN2` / `DLALN2`: `A` and the arithmetic are real, but the
scalar `w` may be complex (then `X` and `B` carry real and imaginary columns). So
the surface is `float` / `double`, constrained by `wwr::real_fp`. The complex
`1x1` and `2x2` divisions go through `calaman::ladiv_scalar` from
`calaman.ladiv`'s header-only, host/device `ladiv.h` — included root-relative by
this module's `.cu` and called per-thread, which is the reason `?ladiv` was built
host/device first (depends on issue #86).

## No argument checking; `info` is the perturbation flag

Like the reference ("in the interests of speed, this routine does not check the
inputs"), there is no bounds checking. `info` is **not** an argument-error code:
it is `0` normally and `1` when `C = ca*op(A) - w*D` had to be perturbed to keep
its smallest singular value above `smin`. The returned `calaman::Status` carries
only the kernel-launch error, if any.

## Shape

`interface.cppm` is the host wrapper (it returns `Status` and re-exports it from
`calaman.error_handling`); `laln2.cu` is the device half — a single-thread port
of reference `?laln2` launched through `wwr.extension.parallel_for`, calling
`ladiv_scalar` for the complex divisions; `laln2_bridge.h` carries the launcher
declaration across the host/device boundary (a global module fragment cannot
`import`). `instantiations.cpp` explicitly instantiates the wrapper for each type.

## Tested

`test/laln2/` is an oracle suite: the device result (`X`, `scale`, `xnorm`,
`info`) must agree with reference LAPACK's `slaln2_` / `dlaln2_` computed on the
host, across both `ltrans` values and every `(na, nw)` shape, plus a
near-singular `info == 1` perturbation case. Reference `?laln2` has no LAPACKE C
binding, so the oracle calls the Fortran symbol directly (as the `lasy2` suite
does). It is `REQUIRES_GPU` (excluded by `ctest -LE gpu`) and builds only when
`calaman::lapack_reference` is present.
