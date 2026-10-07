# calaman.complex_cast

GPU-parallel conversion between real and complex floating-point device arrays.

Four elementwise operations over a device array of `count` elements:

| Operation | Effect | Direction |
|---|---|---|
| `set_real_part` | `output[i] <- (input[i], Im(output[i]))` | complex &larr; real |
| `set_imag_part` | `output[i] <- (Re(output[i]), input[i])` | complex &larr; real |
| `get_real_part` | `output[i] <- Re(input[i])` | real &larr; complex |
| `get_imag_part` | `output[i] <- Im(input[i])` | real &larr; complex |

The two `set_*` operations touch one component and leave the other intact, so a
complex array is **assembled** from two real planes by one `set_real_part` then
one `set_imag_part`, and **split apart** by the two `get_*`. This is the packing
glue a complex solver needs when its inputs or outputs arrive as separate real
and imaginary arrays.

Supported element types: `wwrFloatComplex`, `wwrDoubleComplex` (constrained by
`calaman::complex_fp`). The real component type is `calaman::ComplexToRealType<ComplexT>`
— `float` for `wwrFloatComplex`, `double` for `wwrDoubleComplex`.

Unlike `larfg` / `horner`, complex is the whole point here, not a deferred
extension: there are no `gemm` scalars to spell `constexpr`, only component reads
and writes — which `complex.h`'s accessors (`wwrCreal*`/`wwrCimag*`,
`make_wwr*Complex`) do portably. Raw `.x`/`.y` field access is **not** used:
`cuFloatComplex` is a `float2` but `hipFloatComplex` is a class, so field access
is not portable.

## API

```cpp
import calaman.complex_cast;
import wwr.runtime_api;   // wwrStream_t
import wwr.complex;       // wwrDoubleComplex

template<calaman::complex_fp ComplexT>
void calaman::set_real_part(wwr::wwrStream_t stream, ComplexT* output,
                            const calaman::ComplexToRealType<ComplexT>* input,
                            std::size_t count);

template<calaman::complex_fp ComplexT>
void calaman::set_imag_part(wwr::wwrStream_t stream, ComplexT* output,
                            const calaman::ComplexToRealType<ComplexT>* input,
                            std::size_t count);

template<calaman::complex_fp ComplexT>
void calaman::get_real_part(wwr::wwrStream_t stream,
                            calaman::ComplexToRealType<ComplexT>* output,
                            const ComplexT* input, std::size_t count);

template<calaman::complex_fp ComplexT>
void calaman::get_imag_part(wwr::wwrStream_t stream,
                            calaman::ComplexToRealType<ComplexT>* output,
                            const ComplexT* input, std::size_t count);
```

Each call is one `wwr.extension.parallel_for` pass, enqueued on `stream` and
returning **without synchronizing**. A call with `count == 0` launches nothing.

All pointers are device memory. The operations do **not** alias-check; overlap
other than the deliberate in-place `output` of a `set_*` is the caller's
responsibility.

## Behaviour notes

- **`set_*` is read-modify-write.** There is no portable "set only one field"
  accessor, so each thread reads the element, rebuilds it with the other
  component preserved (`make_wwr*Complex`), and stores it back. The target
  `output` elements must therefore hold valid values on entry if the untouched
  component matters.
- **The four `*_part` calls have no leading-dimension handling.** They are flat
  `count` passes over contiguous arrays. A padded matrix goes through the
  strided pair below.

## Strided matrix pair

```cpp
template<calaman::complex_fp ComplexT>
void calaman::split_planes(wwr::wwrStream_t stream, std::size_t rows, std::size_t cols,
                           const ComplexT* a, std::size_t lda,
                           calaman::ComplexToRealType<ComplexT>* re,
                           calaman::ComplexToRealType<ComplexT>* im, std::size_t ldp);

template<calaman::complex_fp ComplexT>
void calaman::merge_planes(wwr::wwrStream_t stream, std::size_t rows, std::size_t cols,
                           const calaman::ComplexToRealType<ComplexT>* re,
                           const calaman::ComplexToRealType<ComplexT>* im, std::size_t ldp,
                           ComplexT* c, std::size_t ldc);
```

One pass each over a column-major `rows`-by-`cols` block, the complex side with
its own leading dimension and both planes sharing `ldp`; padding rows are
neither read nor written. Generic over the operand: `calaman.lacrm` splits its
first operand A (M-by-N), `calaman.larcm` its second operand B, and both merge
the result planes into C.

## Build

`CMakeLists.txt` builds two targets: `calaman.complex_cast.device` (static,
`complex_cast.cu`, links `wwr.extension.parallel_for`) and the module library
`calaman.complex_cast`. The device archive is pulled in whole
(`$<LINK_LIBRARY:WHOLE_ARCHIVE,...>`) so the explicit `device::*_part`
instantiations survive. Template instantiations are declared `extern template` in
`interface.cppm`, defined once in `instantiations.cpp` (host) and
`complex_cast.cu` (device).

**Dependencies:** `wwr.runtime_api`, `wwr.complex`, `wwr.wrappers.common`, and
`wwr.extension.parallel_for` (device side).

| File | Purpose |
|---|---|
| `interface.cppm` | Module primary interface; the exported wrappers |
| `complex_cast_bridge.h` | Device-launcher declarations, generic in `ComplexT`/`RealT` |
| `complex_cast.cu` | Device functors, launcher definitions, device instantiations |
| `instantiations.cpp` | Module implementation unit; host explicit instantiations |
| `CMakeLists.txt` | Builds the device library and the C++23 module |

## Tests

`test/complex_cast/` (`ComplexCastSpecTests`, REQUIRES_GPU) checks the spec
directly, bit for bit: the flat round-trip, `set_*` preserving the other
component, the `count == 0` no-op, and the strided `split_planes` /
`merge_planes` round-trip with its padding untouched. The `calaman.lacrm` and
`calaman.larcm` oracle suites exercise the strided pair against the reference
LAPACK.
