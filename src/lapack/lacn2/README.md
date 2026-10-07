# `calaman.lacn2`

The GPU counterpart of LAPACK's `?lacn2` (`s`/`d`): Hager's 1-norm estimator
with Higham's refinements, driven by **reverse communication**. The routine
never sees `A`; it asks the caller for products with `A` and `Aᵀ` and returns a
lower bound on `‖A‖₁` — what a matrix-free operator needs (the FEAST driver's
backward-error scale, #304).

```cpp
import calaman.lacn2;   // also re-exports calaman::Status

const std::size_t bytes = calaman::lacn2_bufferSize<double>(n);
double est = 0;
int kase = 0;
std::array<int, 3> isave{};
do {
  calaman::lacn2<double>(blas, n, d_v, d_x, d_work, bytes, est, kase, isave);
  if (kase == 1) { /* d_x <- A * d_x */ }
  else if (kase == 2) { /* d_x <- A^T * d_x */ }
} while (kase != 0);
// est <= ||A||_1;  d_v = A w with est = ||v||_1 / ||w||_1
```

## Mapping from the reference

| `DLACN2` | here |
|---|---|
| `N`, `V`, `X` | `n`, device `d_v`, device `d_x` |
| `ISGN` (workspace) | carved from `d_work` (`lacn2_bufferSize`); pass the same buffer on every call of one run |
| `EST`, `KASE`, `ISAVE` | host references; `isave[1]` stays 1-based as in Fortran |

The state machine runs on the host, exactly as the reference's computed `GO TO`
does. The `?asum`/`i?amax`/`?copy` calls go through the BLAS handle, which must
be in its default **host** pointer mode; the elementwise passes (fill, unit
vector, sign vector, sign comparison, alternating test vector) are kernels in
`lacn2.cu`. Each call synchronizes the handle's stream at most a few times to
read a reduction or an element — the estimator's cost is the caller's products.

The sign test is the reference's `X(I) .GE. ZERO` (LAPACK ≥ 3.10), so a `-0`
maps to `+1` and a NaN to `-1`. Not ported: the complex `?lacn2` (`c`/`z`),
which is a different routine (no `ISGN`, `|x|`-normalised signs).

Tested in `test/lacn2/` against the reference `slacn2_`/`dlacn2_`, called by
Fortran symbol on the same operators: same `kase` sequence, same estimate and
final `v` to the shared tolerance, never above `?lange('1')`, and exact on the
classes where Hager's method is (`n = 1`, diagonal, nonnegative, one dominant
column).
