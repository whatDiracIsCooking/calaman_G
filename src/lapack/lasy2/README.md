# `calaman.lasy2`

The GPU counterpart of LAPACK's `?lasy2`: solve the small Sylvester equation

```
op(TL) * X + ISGN * X * op(TR) = SCALE * B
```

for the `N1`-by-`N2` matrix `X`, with `1 <= N1,N2 <= 2`. `TL` is `N1xN1`, `TR`
is `N2xN2`, `B` is `N1xN2`, `ISGN` is `+1` or `-1`, and `op(T)` is `T` or `Tᵀ`
(selected by `ltranl` / `ltranr`). `SCALE` (`<= 1`) guards the solution against
overflow. One entry point, templated over the two real element types:

```cpp
import calaman.lasy2;      // also re-exports calaman::Status
import wwr.runtime_api;

wwr::wwrStream_t stream{};
wwr::wwrStreamCreate(&stream);
// d_tl, d_tr, d_b: device inputs; d_scale, d_x, d_xnorm, d_info: device outputs
calaman::lasy2<double>(stream, /*ltranl=*/false, /*ltranr=*/false, /*isgn=*/1,
                       /*n1=*/2, /*n2=*/2, d_tl, 2, d_tr, 2, d_b, 2,
                       d_scale, d_x, 2, d_xnorm, d_info);
```

The solve is enqueued on `stream` and returns without synchronizing, like a BLAS
call; the caller synchronizes when it needs the outputs. A stream and not a
device handle, because this routine allocates nothing — it needs neither a device
index nor a memory pool, matching `calaman.lacgv`.

## Everything is on the device

`?lasy2` is called deep inside `?trsyl` / `?trevc` for one `2x2` diagonal block at
a time, where its inputs already live on the device. So the whole algorithm — the
`1x1`, `1x2`, `2x1` and full `2x2` cases, the complete-pivoting `4x4` solve, the
`SMIN` perturbation of near-singular pivots, and the `SCALE` / `XNORM`
computation — runs in a **single device thread**; there is no host round-trip.
That is why the five outputs (`scale`, `x`, `xnorm`, `info`) are device pointers,
not host scalars: nothing is ever read back here.

## Real only

LAPACK ships no complex `?lasy2` (the complex `2x2` Sylvester case is a different
routine), so the surface is `float` / `double`, constrained by `calaman::real_fp` —
the same scope the reference has.

## No argument checking; `info` is the perturbation flag

Like the reference ("in the interests of speed, this routine does not check the
inputs"), there is no bounds checking. `info` is **not** an argument-error code:
it is `0` normally and `1` when `TL` and `TR` had eigenvalues too close, so a
near-singular pivot was perturbed up to `SMIN` to keep the system solvable. The
returned `calaman::Status` carries only the kernel-launch error, if any.

## Shape

`interface.cppm` is the host wrapper (it returns `Status` and re-exports it from
`calaman.error_handling`); `lasy2.cu` is the device half — a single-thread port
of reference `?lasy2` launched through `wwr.extension.parallel_for`;
`lasy2_bridge.h` carries the launcher declaration across the host/device boundary
(a global module fragment cannot `import`). `instantiations.cpp` explicitly
instantiates the wrapper for each type.

## Tested

`test/lasy2/` is an oracle suite: the device result (`X`, `scale`, `xnorm`,
`info`) must agree with reference LAPACK's `slasy2_` / `dlasy2_` computed on the
host, across all `(ltranl, ltranr, isgn)` combinations for each `(n1, n2)` shape,
plus the degenerate quick-return and the near-singular `info == 1` perturbation.
Reference `?lasy2` has no LAPACKE C binding, so the oracle calls the Fortran
symbol directly (as the `larf` suite does). It is `REQUIRES_GPU` (excluded by
`ctest -LE gpu`) and builds only when `calaman::lapack_reference` is present.
