# `test/shared/`

Test-support modules imported by more than one suite. The first two are types
the test tier owns because WarpWraps deliberately stopped shipping them, and
calaman deliberately does not ship them either; the third is the numerical
tolerance the linalg oracle suites share.

| Module | Type |
|---|---|
| `calaman.test.shared.abort_policy` | `calaman::test::AbortPolicy<T>` — print to stderr and abort |
| `calaman.test.shared.device_handle` | `calaman::test::DeviceHandle` — one GPU's index, properties, default stream, default pool |
| `calaman.test.shared.tolerance` | `eps<T>()`, `frobenius_norm`, `factorization_tol` — the shared `O(eps * ‖A‖ * min(m,n))` bound |

## Why they are not in `src/`

WarpWraps used to ship both, as `wwr::extension::AbortPolicy` and a concrete
`wwr::extension::DeviceHandle`. It dropped them on purpose: the concrete handle
hard-codes an abort policy for the stream and pool it owns, so *shipping* the
handle forced an error-handling decision onto every consumer. What it ships now
is the structural `device_handle` / `device_handle_stream` / `device_handle_pool`
concept ladder and the `error_policy` concept — the shapes, not an
implementation.

Re-shipping either one as `calaman::AbortPolicy` or `calaman::DeviceHandle` would
reintroduce exactly that coupling one layer down. So calaman's shipped surface
takes the narrowest thing each routine actually needs:

- `calaman.lacpy` takes a `wwr::wwrStream_t`. It enqueues a kernel and allocates
  nothing, so a stream is the whole requirement.
- `calaman.diff_norm` takes a `wwr::wwrblasHandle_t`, for the same reason.

Neither names a handle type, so neither inherits a policy. The test tier is the
only part of this repo that allocates device memory (`DeviceBufferWrapper` needs
a `device_handle` to pick its allocation strategy), and it is free to choose
abort-on-failure because a failed test fixture *should* die loudly.

## If `src/` ever needs a handle

A routine with a workspace — `?getrf`, `?gesvd` — will need `dev_idx()` to
allocate. When that happens, take the handle as a template parameter constrained
on WarpWraps's concept (`device_handle` or `device_handle_stream`), not as a
concrete class. Note that this trades away the `extern template` firewall that
`calaman.lacpy` uses: an explicit instantiation cannot be written over an open
concept parameter, so such a module either defines its body in the interface
(like `calaman.diff_norm`) or instantiates over a fixed list of handle types.

Do **not** promote `calaman::test::DeviceHandle` into `src/` to avoid that
decision — that is the coupling this file exists to prevent.
