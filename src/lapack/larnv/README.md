# `calaman.larnv`

The GPU counterpart of LAPACK's `?larnv`: fill a device vector from LAPACK's
own random stream, so device-generated test data matches what the reference
would generate from the same `ISEED`.

```cpp
import calaman.larnv;   // also re-exports calaman::Status

// d_iseed: 4 device ints, each in [0, 4095], d_iseed[3] odd; d_x: n values
calaman::larnv<wwr::wwrDoubleComplex>(stream, 4, d_iseed, n, d_x);
```

`IDIST`: 1 uniform (0,1), 2 uniform (−1,1), 3 normal (0,1) by Box–Muller; a
complex `T` adds 4 uniform on the unit disk and 5 uniform on the unit circle.
`IDIST` 1/2 and the updated `ISEED` are bitwise equal to the reference;
`IDIST` 3/4/5 go through the device `log`/`sqrt`/`cos`/`sin`, which are not
bit-equal to libm, so they agree to rounding.

This is not `wwrrandGenerate*`, which is fast but does not produce LAPACK's
sequence. Use this module when the stream itself matters.

## How the chunking stays faithful

The reference walks `x` in chunks of 64 outputs, each one `?laruv` call of
`il2` draws (`2*il` for Box–Muller and for every complex case), threading
`ISEED` through. A `?laruv` call with no retry leaves the seed multiplied by
multiplier row `il2 - 1` mod 2^48, so whole calls compose by multiplication
and chunk `c` starts from `seed · J^c`. `larnv_chunks_kernel` runs every chunk
in parallel from that jump-ahead seed (one block of 128 threads each, through
`laruv_block` from `lapack/laruv/laruv.cuh`).

The one exception is `?laruv`'s retry. In single precision a draw can round to
1.0, and the reference then bumps the seed, which shifts every later chunk.
`larnv_finish_kernel` (one block) scans every chunk's draws in parallel for the
first one that hits 1, replays the walk serially from that chunk with the
threaded seed, and writes the final `ISEED`. In double precision the largest
draw is exactly 1 − 2^-48, so the scan is compiled out and the kernel only
writes the jump-ahead seed. The replay rarely runs: a single draw rounds to 1
with probability about 2^-25.

## Mapping from DLARNV

`s`/`d`/`c`/`z` become one template over `T` (`usual_fp`). An `IDIST` outside
the type's range is an invalid-value `Status`, and so is `n ≥ 2^37`, which
would be more than 2^31 chunks. `n = 0` enqueues nothing and leaves `ISEED`
unchanged.

## Tested

`test/larnv/`: against `LAPACKE_?larnv` for every precision and `IDIST`, n ∈
{1, 2, 63, 64, 65, 128, 200, 2577}, three seeds and chained calls. `IDIST` 1/2
and `ISEED` are compared bitwise, and 3/4/5 to the shared tolerance. Also
single-precision seeds built so a draw in chunk 0, 3 or 17 rounds to 1, for
real `IDIST` 1 and 3 and complex `IDIST` 1. The suite asserts that the
reference really took the retry there.
