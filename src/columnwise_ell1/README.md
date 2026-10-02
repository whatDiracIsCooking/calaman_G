# calaman.columnwise_ell1

The L1 norm of every column of a column-major matrix, in one device kernel:

    d_result[j] = sum_{i=0}^{rows-1} |A(i, j)|   for each column j

This is the per-column form of the `asum` reduction `calaman.diff_norm` wraps as
`Norm::l1`. A single `asum` reduces one vector; `columnwise_ell1` reduces all
`cols` columns in a single launch.

## Shape

Structurally the twin of `calaman.lacpy` — a host module wrapper
(`interface.cppm`) over a device-compiled translation unit (`columnwise_ell1.cu`),
sharing a launcher bridge (`columnwise_ell1_bridge.h`) across the host/device
line, with an explicit-instantiation unit (`instantiations.cpp`). It takes a
`wwr::wwrStream_t`, not a device handle: it enqueues one kernel and allocates
nothing.

The computation is entirely a per-column reduction, so it owns **no kernel of
its own**. `columnwise_ell1.cu` defers to `calaman.reduce_columns` with `|·|` as
the pre-transform and `+` as the fold. Using the *transform* path (not a
fold-in-the-binary-op trick) is what makes a single-row column report `|a|`
rather than `a`.

## Scope

Templated over `float` and `double`, matching the rest of calaman. The L1 norm
of a **complex** column is real-valued (a sum of moduli `sqrt(re² + im²)`), i.e.
a `T → real` reduction rather than this `T → T` one — a deliberate later
extension, as `calaman.diff_norm` notes for the same reason.
`calaman.reduce_columns` is already generic over that differing value type; only
this wrapper is type-fixed.

## Note on the port

Adapted from a CUDA/Thrust reference that threaded an `int* d_keys` scratch
buffer through the API — that was a `thrust::reduce_by_key` artifact. The
hand-written `reduce_columns` kernel needs no such buffer, so it is gone from the
signature.
