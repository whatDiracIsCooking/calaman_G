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
`wwr::complex_fp`). The real component type is `wwr::ComplexToRealType<ComplexT>`
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

template<wwr::complex_fp ComplexT>
void calaman::set_real_part(wwr::wwrStream_t stream, ComplexT* output,
                            const wwr::ComplexToRealType<ComplexT>* input,
                            std::size_t count);

template<wwr::complex_fp ComplexT>
void calaman::set_imag_part(wwr::wwrStream_t stream, ComplexT* output,
                            const wwr::ComplexToRealType<ComplexT>* input,
                            std::size_t count);

template<wwr::complex_fp ComplexT>
void calaman::get_real_part(wwr::wwrStream_t stream,
                            wwr::ComplexToRealType<ComplexT>* output,
                            const ComplexT* input, std::size_t count);

template<wwr::complex_fp ComplexT>
void calaman::get_imag_part(wwr::wwrStream_t stream,
                            wwr::ComplexToRealType<ComplexT>* output,
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
- **No bounds, stride or leading-dimension handling.** These are flat `count`
  passes over contiguous arrays — a column-major matrix with `lda == rows`
  is a contiguous `rows*cols` run and works directly; a padded matrix would
  need a strided variant that does not exist yet.

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
| `interface.cppm` | Module primary interface; the four exported wrappers |
| `complex_cast_bridge.h` | Device-launcher declarations, generic in `ComplexT`/`RealT` |
| `complex_cast.cu` | Device functors, launcher definitions, device instantiations |
| `instantiations.cpp` | Module implementation unit; host explicit instantiations |
| `CMakeLists.txt` | Builds the device library and the C++23 module |

## Tests

Not yet wired. A suite under `test/` would, per element type, round-trip two real
planes through `set_real_part` + `set_imag_part` and back through the two `get_*`,
check that each `set_*` leaves the other component untouched, and check the
`count == 0` no-op — against a host reference, like the other numerical suites.
