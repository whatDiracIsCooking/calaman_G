# `calaman.la_wwaddw`

The GPU counterpart of LAPACK's `?la_wwaddw` (`sla_wwaddw`/`dla_wwaddw`/
`cla_wwaddw`/`zla_wwaddw`): add `w` into a double-word accumulator held as the
pair `(x, y)`, where `x` is the high word and `y` the low word. It works
elementwise over length `n`. Every extra-precise refinement driver
(`?la_{ge,gb,sy,po,he}rfsx_extended`) calls it. It has the
[`calaman.larscl2`](../larscl2/README.md) shape: a stream in, a
`calaman::Status` out, one hand-launched per-element kernel, no workspace and no
`INFO`.

```cpp
import calaman.la_wwaddw;   // also re-exports calaman::Status
// d_x, d_y: device high and low words, length n, updated in place
// d_w: device addends, length n
calaman::la_wwaddw<double>(stream, n, d_x, d_y, d_w);
```

## Mapping from ?LA_WWADDW

The body is the same in all four precisions in the reference. Here it is one
template over `T`, and `N` becomes `std::size_t`. The arrays have no stride, as
in the reference. A complex element runs the real body once per component,
because Fortran's `COMPLEX` `+` and `-` work per component. Unlike the
reference, which checks nothing, a null pointer on a non-empty call returns
`wwrErrorInvalidValue`.

## Every parenthesis is load-bearing

```fortran
S = X(I) + W(I)
S = (S + S) - S
Y(I) = ((X(I) - S) + W(I)) + Y(I)
X(I) = S
```

`(X - S) + W` is the rounding error that `X + W` lost, and it is exact only when
it is evaluated in this order with each operation rounded on its own. An FMA
contraction or a re-association would silently drop it. `(S + S) - S` is an
identity in IEEE binary arithmetic, except when `S + S` overflows. The
reference wrote it to force a rounding on extended-precision registers. It is
kept literally, so that the overflow case matches the reference too.

`la_wwaddw.cu` therefore rounds every add and subtract on its own:

- CUDA uses `__fadd_rn`/`__dsub_rn` and friends, which nvcc never fuses or
  reorders.
- HIP uses plain operators under `#pragma clang fp contract(off)`, because HIP's
  `_rn` functions lower to plain operators that its default `fp-contract=fast`
  then fuses (see [`la_geamv`](../la_geamv/README.md), decision 2).

There are no multiplies to fuse, but the guard costs nothing and keeps the
kernel safe from a later edit.

## Tested

`test/la_wwaddw/` compares `x` and `y` **bit for bit** with a host port of the
Fortran loop, which is compiled under `fp contract(off)` too. A tolerance would
be a false pass here. The output that matters is `y`, which is about one ulp of
`x`, so a port that dropped the compensation would still agree with the
reference to any tolerance scaled to `x`. There is no real oracle: LAPACKE has
no wrapper, `lapack.h` has no `LAPACK_?la_wwaddw`, and Debian's `liblapack`
exports no `?la_wwaddw_`, because the extra-precise routines are built only with
XBLAS.

The cases are `n = 0`, `n = 1`, `w` around one ulp of `x`, `w` cancelling `x`
exactly, the roles swapped, every combination of signed zeros, and `s + s`
overflowing. The swap has two cases. With `|w|` just above `|x|`, the error
survives. With `|w| >> |x|`, the error is lost: the reference's Fast2Sum
assumes `|x| >= |w|`, and the device has to reproduce that loss. A mixed vector of 1000 elements spans several blocks. The
cases that should leave a rounding error also assert that `y` changed. That way
an exact match shows the error term survived, and not just that `y` was left
alone.
