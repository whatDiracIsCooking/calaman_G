# `calaman.lartg`

The device lift of LAPACK's `?lartg`: generate plane rotations. Reference
`?lartg` is a **scalar** routine — one rotation from one pair `(f, g)` — which is
nothing to put on a GPU, so this batches `n` independent rotations, the shape a
caller sweeping a diagonal (a `bdsqr`-style loop) actually wants. One entry
point, templated over the two real element types:

```cpp
import calaman.lartg;     // also re-exports calaman::Status
import wwr.runtime_api;

wwr::wwrStream_t stream{};
wwr::wwrStreamCreate(&stream);
// d_f, d_g: length-n device inputs; d_c, d_s, d_r: length-n device outputs
calaman::lartg(stream, n, d_f, d_g, d_c, d_s, d_r);
```

For each `k` in `[0, n)` it writes `(c[k], s[k], r[k])` such that

```
[ c  -s ] [ f ]   [ r ]
[ s   c ] [ g ] = [ 0 ]
```

with `c >= 0`, `r = sign(f) * hypot(f, g)`, and `c² + s² = 1`. The rotation is
enqueued on `stream` and returns without synchronizing, like a BLAS call; the
caller synchronizes when it needs the outputs. A stream and not a device handle,
because this routine allocates nothing — it needs neither a device index nor a
memory pool, matching `calaman.lacgv` / `calaman.lascl2`.

## Real only

Reference `?lartg`'s complex variants (`clartg` / `zlartg`) carry a **real**
cosine but a **complex** sine and a materially different algorithm, so they are a
deliberate later extension, not a trivial instantiation — the same wall
`calaman.larfg` documents. The surface is `float` and `double`.

## The algorithm, ported exactly

The kernel is a direct per-element port of reference `?lartg`'s safe-scaling
algorithm (Anderson, *Algorithm 978: Safe Scaling in the Level 1 BLAS*, 2017):

- `g == 0` → `c = 1, s = 0, r = f`;
- `f == 0` (and `g ≠ 0`) → `c = 0, s = sign(1, g), r = |g|`, no floating-point
  work;
- both magnitudes inside `[rtmin, rtmax]` → the unscaled fast path
  (`d = hypot(f, g)` directly);
- otherwise → scale by `u = min(safmax, max(safmin, |f|, |g|))` first, so the
  sum of squares neither overflows nor underflows.

The four thresholds are precision constants — `safmin` the smallest normal,
`safmax = 1/safmin`, `rtmin = sqrt(safmin)`, `rtmax = sqrt(safmax/2)` — computed
once on the host in the launcher and handed to every thread. `fabs`, `sqrt`,
`copysign`, `fmin` and `fmax` go through `wwr.wrappers.math`
(`wrappers/math/math.cuh`), which spells the `f`-suffixed intrinsic once per
type; the kernel never names `::sqrtf` vs `::sqrt`. `copysign(a, b)` is LAPACK's
`SIGN(a, b)` and is only reached where the sign source is non-zero, so IEEE's
signed-zero rule never diverges from Fortran's.

## Mapping from `?lartg`

Changed, per `docs/architecture.md` §4: the scalar arguments `(F, G, C, S, R)`
become the per-element device arrays, and the batch count `n` is `std::size_t`.
The three output arrays must be **distinct** from one another; an output may
alias an input (each thread reads `f`/`g` before writing `c`/`s`/`r`). There is
no `INFO` — `?lartg` reports none; this wrapper returns `calaman::Status`,
carrying the kernel-launch error if one occurs. `n == 0` is a no-op success.

## Shape

`interface.cppm` is the host wrapper (it returns `Status` and re-exports it from
`calaman.error_handling`); `lartg.cu` is the device half — a single per-element
functor launched through `wwr.extension.parallel_for`; `lartg_bridge.h` carries
the launcher declaration across the host/device boundary (a global module
fragment cannot `import`). `instantiations.cpp` explicitly instantiates the
wrapper for each type.

## Tested

`test/lartg/` is an oracle suite: the device result must agree with reference
LAPACK's `slartg` / `dlartg` to the shared tolerance. Because LAPACKE exposes no
`?lartg` C wrapper (only `lartgp` / `lartgs`), the oracle calls the Fortran
symbols `slartg_` / `dlartg_` directly — the exact routine the kernel ports —
reached through `calaman::lapack_reference` (which links `LAPACK::LAPACK`). The
cases cover both signs, the `g == 0` and `f == 0` branches, the subnormal and
overflow-magnitude inputs that exercise the scaled fallback, and the `n == 0`
no-op. It is `REQUIRES_GPU` (excluded by `ctest -LE gpu`) and builds only when
`calaman::lapack_reference` is present.
