# calaman.set_element

Read **one** element of a column-major/strided device vector, selected by an
index that itself lives on the device, and write it (or its magnitude) to a
device scalar. One directory, one kernel, the laset/complex_cast module shape:
`interface.cppm` + `set_element.cu` + `set_element_bridge.h` + an
`instantiations.cpp` explicit-instantiation unit. Built for either backend from
one source.

## Why it exists

A cuBLAS/hipBLAS handle has two *pointer modes*. In **host** pointer mode a
reduction like `I?AMAX` writes its result to host memory and the call
synchronizes, so the host can immediately use it — this is what
`calaman.diff_norm`'s ell_inf path relies on: iamax hands back a host `int`
index, and the host reads that element and takes its magnitude.

In **device** pointer mode iamax writes its 1-based index to a *device* `int`
(allocated on the handle's stream, reachable via `wwrblasGetStream`), and the
result scalar is a device pointer too. There is **no** vendor BLAS call that
turns that device index into the element's value. That gap is this module:

- `set_element_abs<T>(stream, d_x, incx, d_idx, d_result)` — writes
  `|d_x[(*d_idx - 1) * incx]|`, the max-magnitude **value** iamax never gives
  you, entirely on the device. The output is real, so `d_result` is
  `wwr::ComplexToRealType<T>` (which is `T` for a real element).
- `set_element<T>(stream, d_x, incx, d_idx, d_result)` — the same indexed read
  **without** the `|·|`: the generic "gather the element at a device-computed
  index" primitive. Output type `T`.

`d_idx` holds a **1-based** index, matching iamax's Fortran convention; the
kernel subtracts 1. All of `d_x`, `d_idx`, `d_result` are device memory. Both
calls are asynchronous (enqueue one item on `stream`, no synchronize) and assume
a valid in-range element exists — an empty vector has no index to read, so the
caller must guard that case (as diff_norm guards `n == 0`).

## One kernel, a unary functor

The two operations differ only in a per-element transform, so `set_element.cu`
carries one `SetElementFunctor` that does the device-index read and defers the
transform to a `UnaryOp` template argument: gather is `IdentityOp`, magnitude is
`AbsOp` over `calaman::device::elem_ops<T>::modulus` (the one portable `|·|`
across `float`, `double`, and the two complex types — never `.x`/`.y`, which
`hipComplex` lacks). A third transform would be one more functor plus its
launcher, no new kernel. The single element is set through one
`wwr.extension.parallel_for` item — the outermost WarpWraps launch layer.

## Types

`float`, `double`, `wwrFloatComplex`, `wwrDoubleComplex` (constrained
`wwr::usual_fp`). For a complex element, `set_element` gathers the complex value
and `set_element_abs` writes its real modulus.

## Consumers

`calaman.diff_norm` — its ell_inf norm uses `set_element_abs` in device pointer
mode: `iamax` writes its index to the device, and this turns it into the
max-magnitude value at the device `result`, with no host round trip.
