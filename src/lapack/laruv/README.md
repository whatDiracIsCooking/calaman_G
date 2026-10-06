# `calaman.laruv`

The GPU counterpart of LAPACK's `?laruv`: up to 128 uniform (0,1) draws from
the 48-bit multiplicative congruential generator that `?larnv` (and so every
LAPACK test-matrix generator) is built on, reproduced **bit for bit** — the
draws and the updated `ISEED`.

```cpp
import calaman.laruv;   // also re-exports calaman::Status

// d_iseed: 4 device ints, each in [0, 4095], d_iseed[3] odd; d_x: n <= 128
calaman::laruv<double>(stream, d_iseed, n, d_x);
```

The call is enqueued on `stream` and returns without synchronizing; the seed is
read and then overwritten on the device, so consecutive calls on one stream
continue LAPACK's stream exactly as consecutive `dlaruv` calls do.

## Why it is bit-exact

Draw `i` is the seed times the multiplier row `MM(i)` (four 12-bit limbs),
mod 2^48, computed in `int` limb arithmetic exactly as the reference does, then
scaled by `R = 2^-12` per limb. `R` is a power of two, so every `R*y` is exact
and the only roundings are the three additions — the same ones the reference
makes, which FMA contraction cannot change. In single precision a draw can
round to `1.0`; the reference then adds 2 to every seed limb and redraws, and
that bump persists for the later draws. `laruv_block` reproduces it by
re-running the tail of the block from the first index that hit.

## Shape

- `laruv.cuh` — the multiplier table (`__constant__`), `laruv_draw` (one
  output) and `laruv_block` (the block-cooperative call). Linked as the
  INTERFACE target `calaman::laruv::header`, for `?larnv`'s kernels.
- `laruv.cu` — one block of 128 threads calling `laruv_block`.
- `laruv_bridge.h`, `interface.cppm`, `instantiations.cpp` — the usual
  module / `.cu` split (`calaman.lacpy`'s shape).

## Mapping from DLARUV

`s`/`d` become one template over `T` (`real_fp`); LAPACK has no complex
variant. `N > 128`, a documented precondition of the reference, is an
invalid-value `Status`; `N = 0` enqueues nothing and leaves `ISEED` unchanged.

## Tested

`test/laruv/`: bitwise against the Fortran `slaruv_`/`dlaruv_` (LAPACKE has no
binding) for a spread of seeds and n ∈ {1, 2, 64, 127, 128}, chained calls, and
single-precision seeds constructed so a draw rounds to 1.0 at index 0, 5 and 100.
