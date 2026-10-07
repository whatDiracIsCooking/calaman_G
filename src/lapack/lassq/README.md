# `calaman.lassq`

The GPU counterpart of LAPACK's `?lassq`: fold the sum of squares of a strided
vector into a caller-held `(scale, sumsq)` pair, so that on return

```
scale^2 * sumsq  ==  sum_i |x(i)|^2  +  scale_in^2 * sumsq_in
```

without ever squaring an entry — or the running total — outside the exponent
range. One entry point, templated over `float`, `double`, `wwrFloatComplex` and
`wwrDoubleComplex` (`?lassq`'s `s`/`d`/`c`/`z`), with both scalars in `T`'s real
component type:

```cpp
import calaman.lassq;    // also re-exports calaman::Status
import wwr.runtime_api;

// d_scale, d_sumsq: device scalars. (1, 0) starts a fresh accumulation.
calaman::lassq<double>(stream, n, d_x, 1, d_scale, d_sumsq);
```

The pair is read-modify-written on the device and the call returns without
synchronizing. `d_x`, `d_scale` and `d_sumsq` are device pointers the caller
owns; nothing is allocated here, so a stream — not a device handle — is the
whole requirement, matching `calaman.lanst`. See `test/shared/README.md`.

## Why a pair and not a norm

An accumulating `(scale, sumsq)` is the point of `?lassq`, not an awkwardness of
its Fortran signature: it is how LAPACK builds a Frobenius norm one column at a
time, each call folding another column into the same pair. `calaman.lange` and
the other `lan*` norms currently take the plain-sum shortcut instead (their
headers say so); this module is the overflow-safe accumulator they would need to
drop it.

## The three accumulators

`?lassq` (LAPACK ≥ 3.10) splits the entries into three bands by magnitude and
scales each band before squaring, so no square over- or underflows:

| Band | Accumulator | Scaled by |
|---|---|---|
| `\|x\| > tbig` | `abig` | down, by `sbig` |
| mid-range | `amed` | not at all |
| `\|x\| < tsml` | `asml` | up, by `ssml` |

The four constants are exact powers of two under IEEE-754, written out as hex
float literals in `lassq.cu` (a device pass has no `<limits>`) with the
radix-model exponent each comes from, from LAPACK's `la_constants.f90`.

Each thread folds a contiguous chunk into its own three accumulators;
`common/block_reduce.cuh` sums each across the block, and thread 0 runs
`?lassq`'s combine step and writes the pair back. One block, as in
`calaman.lanst`: the fold is memory-bound over a single vector, and one block
read-modify-writes the pair with no second pass and no grid-wide barrier.

**`notbig` is `abig == 0`.** The reference carries a sequential flag that stops
feeding `asml` once an entry above `tbig` has been seen; a parallel fold has no
"once". It needs none: `asml` is read only in the combine's `else if (asml > 0)`
arm, which `abig > 0` already excludes, and `abig > 0` holds exactly when some
entry exceeded `tbig`. The result is the reference's.

A complex entry contributes `|Re|` and `|Im|` as two separate entries, as
`ZLASSQ`'s loop does.

## Mapping from DLASSQ

Kept: the name, the accumulating pair, and the signed `INCX` — a negative stride
walks the same elements from the far end, as `?lassq`'s `ix` does. Changed, per
`docs/architecture.md` §4: the four variants become one template over `T`, and
`SCALE`/`SUMSQ` become device pointers. There is no `INFO` — `?lassq` reports
none.

Faithful at the edges: a NaN in either scalar on entry leaves both untouched,
`n <= 0` applies only the canonicalization of a pair representing zero (so the
kernel still launches for it), and `incx == 0` reads `x[0]` `n` times, which is
what the reference's `ix` stepping does.

Sums are folded in a different order than the reference's single loop, so the
device result agrees to rounding, not bit-for-bit.

## Shape

`interface.cppm` is the host wrapper; `lassq.cu` is the device half;
`lassq_bridge.h` carries the launcher declaration across the host/device
boundary (a global module fragment cannot `import`). `instantiations.cpp`
explicitly instantiates the wrapper for each type.

A per-thread `CLM_HOST_DEVICE` helper — the `lapack/lanst/lanst.h` shape, for a
kernel that needs `?lassq` of a sub-range from one thread — is the natural
extension and is not here yet; nothing in the tree calls for it.

## Tested

`test/lassq/` runs the kernel on the device and compares against the reference
`LAPACK_?lassq` (declared by `lapack.h`, which `lapacke.h` includes; LAPACKE
wraps no `?lassq`) over the identical inputs. The comparison is of the value the
pair *represents*, computed in `long double` so an input deliberately near the
overflow or underflow threshold can still be compared on the host — the pair's
own normalization is an implementation choice, the represented value is the
contract. The suite is `REQUIRES_GPU` (labeled `gpu`, excluded by
`ctest -LE gpu`).
