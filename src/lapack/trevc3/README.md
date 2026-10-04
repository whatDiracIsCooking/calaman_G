# `calaman.trevc3`

The GPU counterpart of LAPACK's `?trevc3`: compute the left and/or right
eigenvectors of an upper quasi-triangular (real Schur) matrix `T` by
back-substitution. Each eigenvalue — real (a `1x1` diagonal block) or a
complex-conjugate pair (a `2x2` block) — yields one column (real) or two columns
(real part, then imaginary part) of the eigenvector matrix, in the diagonal order
of `T`, each normalized so its largest-magnitude entry is `1`. Real element type,
templated over the two precisions:

```cpp
import calaman.trevc3;     // also re-exports calaman::Status
import wwr.runtime_api;

wwr::wwrStream_t stream{};
wwr::wwrStreamCreate(&stream);
// d_t: N x N real Schur form; d_vr: N x N output; d_work: length-3N scratch
calaman::trevc3<double>(stream, /*want_left=*/false, /*want_right=*/true, n,
                        d_t, ldt, /*vl=*/nullptr, 1, d_vr, ldvr, d_work);
```

The computation is enqueued on `stream` and returns without synchronizing, like a
BLAS call. A stream and not a device handle, because this routine allocates
nothing — the caller provides the length-`3N` `work` scratch — matching
`calaman.laqr5`.

## `HOWMNY = 'A'`, eigenvectors of `T` directly

The reference `?trevc3` has three jobs selected by `HOWMNY`: `'A'` all
eigenvectors, `'S'` a `SELECT`-ed subset, and `'B'` all eigenvectors
back-transformed onto an input matrix `Q` (the `OVER` path, which is where the
blocked `?trevc3` differs from the unblocked `?trevc` — it batches the
back-transform into `DGEMM` column-block multiplies). This module exposes only
`'A'` with the eigenvectors written straight into `VL` / `VR`: `SELECT` is unused
and the output count `M` is always `N`. That is the mode whose oracle comparison
is clean, and the one the `geev` driver (issue #103) composes with an explicit
back-transform of its own.

## Everything is on the device, in one thread

`?trevc3`'s back-substitution is inherently sequential — each eigenvector's
components are solved from the diagonal inward, reusing the ones already found —
so the whole routine runs in a **single device thread**: the real/complex split
off the subdiagonal, the right-vector downward sweep (`KI = N..1`), the
left-vector upward sweep over the transposed trailing block (`KI = 1..N`), the
per-block scaled solve, the overflow rescaling, and the final infinity-norm
normalization. The per-block work is `O(1)`; each back-substitution update
touches `O(N)` entries of the growing vector in that same thread. There is no host
round-trip, so `vl` / `vr` / `work` are device pointers.

## `?laln2` is inlined, not imported

Each diagonal block needs the scaled `1x1` / `2x2` solve LAPACK calls `?laln2`
(`calaman.laln2`). That module's device half is a `parallel_for` **launch
functor**, not a callable `__device__` helper, so it cannot be invoked from inside
this kernel. As `calaman.laexc` inlines its `?lartg` / `?lasy2` / `?lanv2` /
`?larfg` auxiliaries, `trevc3.cu` inlines a `__device__` `?laln2` solve — the same
arithmetic `src/laln2/laln2.cu` ships, line for line (that module's oracle pins it
to the reference). Its complex divisions reach the **same** header-only
host/device `calaman::ladiv_scalar` from `calaman.ladiv`'s `ladiv.h` the reference
reaches as `DLADIV`, included root-relative (`"lapack/ladiv/ladiv.h"`) so the call has
device linkage. The kernel therefore links no sibling LAPACK module.

## Normalization fixes the sign

An eigenvector is defined only up to a scalar; `?trevc3` pins it by the
infinity-norm convention — scale so the largest-magnitude entry is exactly `1`
(for a complex pair, the largest `|re| + |im|` over the rows). Both the device and
the reference use this identical rule, so the two results agree up to the sign the
convention leaves — which is why the oracle test compares a sign-invariant
residual rather than entry-for-entry.

## No argument checking

Like the reference in its computational path, there is no bounds checking. The
returned `calaman::Status` carries only the kernel-launch error, if any.

## Shape

`interface.cppm` is the host wrapper (it returns `Status` and re-exports it from
`calaman.error_handling`); `trevc3.cu` is the device half — a single-thread port
of reference `?trevc3` (`HOWMNY = 'A'`) launched through
`wwr.extension.parallel_for`, with the inlined `?laln2` solve calling
`ladiv_scalar`; `trevc3_bridge.h` carries the launcher declaration across the
host/device boundary (a global module fragment cannot `import`).
`instantiations.cpp` explicitly instantiates the wrapper for each type.

## Tested

`test/trevc3/` is an oracle suite: the device eigenvectors must agree with
reference LAPACK's `strevc3_` / `dtrevc3_` computed on the host, for both `s` and
`d`, over right-only, left-only and both, and over matrices with real,
complex-conjugate, and mixed eigenvalues. `?trevc3` has no LAPACKE C binding, so
the oracle calls the Fortran symbol directly (as the `laqr5` suite does).
Eigenvectors match only up to sign/scale, so the suite checks a scale/sign-
invariant residual (`‖T v − λ v‖` for right vectors, the analogous left relation)
against `test/shared/tolerance.cppm`. It is `REQUIRES_GPU` (excluded by `ctest -LE
gpu`) and builds only when `calaman::lapack_reference` is present.
