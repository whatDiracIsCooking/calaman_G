# `calaman.test.elementwise_compare`

A GPU test utility for comparing two device arrays, so a suite checking a
computed device result against an expected one gets its answer without copying
the whole array to the host and looping there. Three entry points, all
device-pointers-in / scalar-out:

- **`count_mismatches(handle, a, b, n) -> unsigned int`** — how many elements
  differ by **exact inequality** (`a[i] != b[i]`). For bit-identical checks:
  determinism, two GPU runs, integer results.
- **`count_beyond_tolerance(handle, a, b, n, atol, rtol = 0) -> unsigned int`** —
  how many elements exceed `atol + rtol*|b[i]|`. The float-friendly counterpart,
  for a result-vs-reference check where exact equality never holds.
- **`max_abs_diff(handle, a, b, n) -> T`** — the worst `max|a[i]-b[i]|`. The
  magnitude metric: assert it is at most the tolerance a comparison allows.

All are templated over `float` and `double`. `handle` is a
`std::shared_ptr<calaman::test::DeviceHandle>` — a device handle rather than a
bare stream, because each of these allocates a scratch buffer and so needs the
device index and memory pool a stream does not carry. That is the opposite of
`calaman.lacpy`, which allocates nothing and takes a `wwrStream_t`; see
`test/shared/README.md`.

## Why two kernels

Each answer is produced wholly on the device, in two stages (both in
`elementwise_compare.cu`):

1. **A per-element map** writes one scratch value per element -- an exact flag
   (`write_mismatch_flags`), a tolerance flag (`write_tolerance_flags`), or the
   absolute difference (`write_abs_diff`). These are index-per-thread, so they
   reuse `wwr.extension.parallel_for` rather than hand-written kernels.
2. **A single-block reduction** collapses the scratch buffer to one scalar:
   `reduce_count` sums the 0/1 flags, `reduce_max` takes the maximum. One block
   keeps the result a single scalar with no inter-block second pass; the gpu\*
   layer has no reduction primitive of its own.

## Shape

`interface.cppm` is the host wrapper; `elementwise_compare.cu` is the device
half; `elementwise_compare_bridge.h` carries the device-launcher declarations
across the host/device boundary (a global module fragment cannot `import`).
`instantiations.cpp` explicitly instantiates each exported wrapper for `float`
and `double`, paired with the `extern template` list in `interface.cppm`; the
`.cu` instantiates the device launchers for the same types. Add a type by
extending every list together. This mirrors WarpWraps's `wwr.extension.random_normal`.
The self-test lives at `test/test/elementwise_compare/`.
