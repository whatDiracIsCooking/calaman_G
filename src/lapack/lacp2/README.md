# `calaman.lacp2`

The GPU counterpart of LAPACK's `?lacp2`: copy all, the upper, or the lower
triangle of a real column-major matrix `A` into a complex matrix `B`, with
imaginary part 0. One entry point, templated over the complex type:

```cpp
import calaman.lacp2;   // also re-exports calaman::Region and Status

// d_a: m x n double (lda); d_b: m x n wwrDoubleComplex (ldb)
calaman::lacp2<wwr::wwrDoubleComplex>(stream, calaman::Region::U, m, n, d_a, lda,
                                      d_b, ldb);
```

The surface is `calaman.lacpy`'s: a `Region`, a bare `wwrStream_t`, nothing
allocated, no synchronization, and every element of `B` outside the copied
region (including the `ldb - m` padding rows) left untouched. The complex type
is constrained by `calaman::complex_fp` and the source's element type is
`calaman::ComplexToRealType<ComplexT>`, as in `calaman.complex_cast`.

`complex_cast::set_real_part` is not a substitute: it works on a flat array (no
region, no leading dimensions) and keeps the existing imaginary part.

## Shape

`lacp2.cu` is `lacpy.cu`'s kernel with a widening store: one thread per element
over a 2-D grid (`4*WWR_WARP_SIZE` threads along the rows, one block per column
in y), the element built with `make_complex` from `common/elem_ops.cuh` so no
`.x`/`.y` access leaks a backend. `lacp2_bridge.h`, `interface.cppm` and
`instantiations.cpp` are the usual module / `.cu` split.

## Mapping from ZLACP2

`UPLO` becomes `calaman::Region`, `c`/`z` become one template over `ComplexT`,
the extents become `std::size_t`, and `LDA`/`LDB` stay. There is no `INFO`.

## Tested

`test/lacp2/`: bitwise against `LAPACKE_clacp2`/`LAPACKE_zlacp2` for every
`Region` over square, tall, wide and 1x1 shapes with `lda > m` and `ldb > m`,
comparing the whole sentinel-filled `ldb`-by-`n` buffer, plus the m = 0 / n = 0
no-op.
