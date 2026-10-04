# calaman.inv_sqrt

Symmetric (Loewdin) inverse square root `X = S^{-1/2}` of a real symmetric
positive-(semi)definite matrix, over the WarpWraps `syevd` + `dgmm`/`gemm`
wrappers and a small `inverse_sqrt` device kernel.

## Module

`calaman.inv_sqrt`

## Design

Given a real-symmetric, positive-(semi)definite operand `S` (column-major,
packed leading dimension `n`), it builds

```
X = S^{-1/2} = U * Lambda^{-1/2} * U^T
```

where `S = U Lambda U^T` is the symmetric eigendecomposition (columns of `U` are
eigenvectors, `Lambda` the ascending eigenvalues). This is the Loewdin
orthogonalizer: it turns a generalized eigenproblem `F C = S C eps` into a
standard one (`F' = X^T F X`, `C = X C'`), and the same `S^{-1/2}` shape recurs
for any symmetric PSD operand — an overlap, a density-fitting Coulomb metric. It
is the symmetric cousin of `calaman.orthogonalize` (that one is the QR thin-Q of
a *general* matrix; this one is the eigen-root of a *symmetric* one).

The computation, all on the GPU over the outermost WarpWraps wrappers:

1. `syevd(S)` → ascending eigenvalues `lambda_j` and eigenvectors `U`. `syevd`
   overwrites its input, so `S` is first copied into the caller's `u_scratch` —
   `s_dev` is left untouched.
2. `inverse_sqrt` kernel: `Lambda <- diag(lambda_j^{-1/2})`, in place, with modes
   at or below `eigenvalue_floor` (default `1e-10`) set to 0 (see Conditioning).
3. `M = U * diag(Lambda^{-1/2})` (`dgmm`, `side = RIGHT`: column scaling).
4. `X = M * U^T` (`gemm`, `transb = T`). `X` is symmetric by construction.

It **allocates nothing**: every intermediate — `U`, `M`, the eigenvalue buffer,
the `syevd` workspace, the convergence flag — is caller-owned and passed in
explicitly. `inv_sqrt_bufferSize` sizes the `syevd` workspace.

**Conditioning.** The `inverse_sqrt` kernel *floors*: an eigenvalue at or below
`eigenvalue_floor` contributes 0 instead of `lambda_j^{-1/2}`, dropping that
mode. Because the check happens before the square root, `X` is free of NaN/Inf
even for a zero or negative eigenvalue. The decision is per-element in the
kernel, so steps 2–4 are **fully stream-ordered — no sync, no host round-trip.**
`syevd`'s convergence flag is handed back to the caller in `info_device` to
inspect, rather than read back internally (the `calaman.orthogonalize`
convention). Consequently `X^T S X == I` holds only when every eigenvalue clears
the floor (a positive-definite, well-conditioned `S`); for a singular or
indefinite `S` the dropped modes make `X^T S X` a rank-deficient projector, which
is expected since such an `S` has no true `S^{-1/2}`. This is **not** full
canonical orthogonalization — `X` stays a square `n x n` matrix.

## Port notes

Ported from a cuSOLVER/cuBLAS-specific routine. Two things changed to fit
calaman: the raw `cusolver`/`cublas` calls became the backend-neutral `wwr*`
wrappers, and the original's internal host round-trip (which both checked
`syevd` convergence and logged `cond(S)`) was dropped — calaman ships no logger,
and its convention is to hand the `info` flag back to the caller, which also
keeps the routine fully stream-ordered.

## Files

| File | Role |
|------|------|
| `interface.cppm` | Module interface — `inv_sqrt` + `inv_sqrt_bufferSize` |
| `instantiations.cpp` | Explicit `float`/`double` instantiations |
| `inv_sqrt.cu` | The `inverse_sqrt` device kernel (over `parallel_for`) |
| `inv_sqrt_bridge.h` | GMF-shared `inverse_sqrt` launcher declaration |
| `CMakeLists.txt` | Build configuration |
