# calaman.elem_ops

Header-only, backend-neutral per-element scalar arithmetic over the four element
types (`float`, `double`, `wwrFloatComplex`, `wwrDoubleComplex`). One directory,
one header (`elem_ops.cuh`), no module and no `.cu` of its own: consumers
`#include` it into their own device translation unit, like
`calaman.reduce_columns` and WarpWraps' `wwr.extension.parallel_for`.

## The problem it solves

Device code templated on the scalar type cannot write `a + b` across both
families. A vendor complex is an operator-less `float2` on CUDA but a class on
HIP, so operators and brace-init are not portable; WarpWraps' `complex.h`
therefore exposes arithmetic only as C-style `wwrCadd`/`wwrCmul`/… functions and
**refuses to overload operators** on types it does not own. And the two families
carry different-shaped portable vocabularies:

| | real (`float`/`double`) | complex (`wwr*Complex`) |
|---|---|---|
| `+ - * /` | language operators | `wwrCadd/wwrCsub/wwrCmul/wwrCdiv` |
| `exp/cos/sin/…` | `wwr::exp` … (`math.cuh`) | **nothing** — `math.cuh` `static_assert`s float/double |

`elem_ops<T>` is the seam that gives both one name, so a kernel body reads the
same regardless of `T`.

## API

`calaman::device::elem_ops<T>` — a trait of `__device__ __forceinline__` static
functions. Primary template = the real types; explicit specializations =
the two complex types (one token-pasting macro stamps both, so the precisions
cannot drift).

- `using real_type` — `T` for reals, the component type for complex.
- `zero()`, `one()`, `from_real(r)` — build an element from a real.
- `add/sub/mul/div(a, b)` — the four arithmetic ops.
- `conj(a)` — conjugate (identity for reals).
- `scale(a, s)` — multiply by a **real** scalar `s` (no complex temporary).
- `real_part(a)` → `real_type`, `modulus(a)` → `real_type` (the true `|z|`).
- `exp(a)` — forwards to `math.cuh` for reals; for complex, hand-built as
  `exp(re)·(cos im, sin im)` from the real `wwr::exp/cos/sin`, since `math.cuh`
  has no complex spelling.

`make_complex(re, im)` is a complex-only free overload set beside the trait (no
real counterpart — a real "complex from two components" is meaningless), the one
place `make_wwr*Complex` is selected by precision.

## What stays out

The **core** only. A module-specific op layers on top as a free `__device__`
helper over the core — never a member here — so a consumer never instantiates
ops it does not call. See `src/expm/expm.cu`'s `fma_real` / `add_real` (the
real-coefficient fused multiply-add the Padé ladder needs) and `src/lapack/gebal/gebal.cu`'s
`abs1` (the CABS1 surrogate `|Re| + |Im|`) / `is_zero` — each one or two lines
over `elem_ops::add` / `scale` / `from_real` / `real_part` / `imag_part`.

## Consumers

- `src/expm` — the matrix exponential's fused Padé kernels (core arithmetic).
- `src/lapack/gebal` — balancing: `scale`, plus `abs1` / `is_zero` as local helpers.
- `src/complex_cast` — component packing: `real_part` / `imag_part` / `make_complex`.

New device kernels that need type-generic scalar arithmetic should link
`calaman.elem_ops` and reach for this rather than re-rolling a per-type
`elem_ops` — the pattern `expm`, `gebal` and `complex_cast` each hand-copied
before this header existed.
