# `calaman.rscl`

The GPU counterpart of LAPACK's `?rscl`: multiply a strided vector by the
reciprocal of a real scalar — `x <- x / a` — without over- or underflow, for any
`a` such that `x/a` is itself representable. One entry point, templated over
`float`, `double`, `wwrFloatComplex` and `wwrDoubleComplex`, with the scalar in
`T`'s real component type:

```cpp
import calaman.rscl;     // also re-exports calaman::Status
import wwr.runtime_api;

calaman::rscl<double>(stream, n, a, d_x, 1);               // d_x <- d_x / a
calaman::rscl<wwr::wwrDoubleComplex>(stream, n, a, d_z, 1); // zdrscl
```

The scaling is enqueued on `stream` and returns without synchronizing, like a
BLAS call; the caller synchronizes when it needs `x`. `x` is a device pointer
the caller owns; nothing is allocated here, so a stream — not a device handle —
is the whole requirement, matching `calaman.lascl`. See
`test/shared/README.md`.

## Why not just multiply by `1/a`

Forming `1/a` first loses the routine's whole guarantee twice over: for a tiny
`a` the reciprocal overflows to infinity, and for a huge `a` it underflows to
zero or a subnormal, so `x * (1/a)` is zero where `x/a` is perfectly
representable. `?rscl` instead carries the fraction as a separate numerator and
denominator, shifting them by `smlnum` and `bignum` until their quotient is in
range, and applies the shift to `x` as it goes. Each factor it hands the vector
is guaranteed in range, so no intermediate product leaves the exponent range.

That loop is host scalar arithmetic and lives in `interface.cppm`, verbatim from
netlib; the device sees only one real factor per iteration — the role the
reference gives `?scal`. It is the same decomposition `calaman.lascl` runs for
`cto/cfrom`, which is `?rscl`'s own loop with `cto = 1`. Typically one factor
and so one launch; two or three in the extreme cases the loop exists for.

## Scope

The **real-scalar** family, which is four of LAPACK's six: `srscl`, `drscl`
(real vector) and `csrscl`, `zdrscl` (complex vector, real scalar) — one
template over `T`, since `elem_ops<T>::scale` takes a real factor for either
family.

The **complex-scalar** `crscl` / `zrscl` are *not* implemented. They are not a
type extension of this code: a safe complex reciprocal is a different algorithm
(the reference splits on which component is zero and guards against the
intermediate `|a|^2` over- or underflowing), and it belongs in its own module
beside `calaman.ladiv`, which already owns safe complex division.

## Mapping from DRSCL

Kept: the name, the multiplier loop, and the signed `INCX`. Changed, per
`docs/architecture.md` §4: the four variants become one template over `T`.

Faithful at the edges: `n < 1` is a no-op, and so is `incx <= 0` — the
reference's `?scal` returns early for a non-positive stride, which makes the
whole call a no-op however the multiplier chain runs.

**One deliberate divergence.** `a == 0` or a non-finite `a` returns
`InvalidValue` rather than being passed through. The reference reports no `INFO`
and, on an infinite `a`, *spins forever*: `cden = inf` makes `cden * smlnum`
still infinite, so the loop never reaches `done`. `a == 0` is outside `?rscl`'s
stated contract as well, since `x/0` is not representable. Rejecting both is the
same choice `calaman.lascl` makes for `cfrom == 0` (`?lascl`'s `INFO = -4`).

## Shape

`interface.cppm` is the host wrapper and owns the multiplier loop; `rscl.cu` is
one per-element functor launched through `wwr.extension.parallel_for` —
`calaman.lacgv`'s shape; `rscl_bridge.h` carries the launcher declaration across
the host/device boundary (a global module fragment cannot `import`).
`instantiations.cpp` explicitly instantiates the wrapper for each type.

**Why its own kernel, when `wwr.wrappers.blas` exports `scal`.** `wwr::scal`
would be the outer layer, and its `Cs`/`Zd` dispatch is exactly the
`csscal`/`zdscal` the reference's complex variants call — but it takes a
`wwrblasHandle_t`, and with it that handle's pointer mode and error policy.
`?rscl` allocates nothing and needs no pool, so the narrowest handle that does
the job is a bare stream (`test/shared/README.md`), which is also what
`calaman.lascl` takes for the identical multiplier loop. One per-element functor
is the price of keeping the signature that narrow.

## Tested

`test/rscl/` runs the kernel on the device and compares against the reference
`?rscl` over the identical inputs. LAPACKE wraps no `?rscl` and neither does
`lapack.h`, so the suite declares the four Fortran symbols (`srscl_`, `drscl_`,
`csrscl_`, `zdrscl_`) itself — `test/lartg/`'s precedent. Both paths apply the
*same* chain of factors in the same order, so the comparison is bit-for-bit
`==`, including the cases chosen so the chain takes more than one step. The
suite is `REQUIRES_GPU` (labeled `gpu`, excluded by `ctest -LE gpu`).
