# calaman.columnwise_ell2

The L2 (Euclidean) norm of every column of a column-major matrix, in two device
passes:

    d_result[j] = sqrt(sum_{i=0}^{rows-1} A(i, j)^2)   for each column j

This is the per-column form of the `nrm2` reduction `calaman.diff_norm` wraps as
`Norm::l2`. A single `nrm2` reduces one vector; `columnwise_ell2` reduces all
`cols` columns together. It is the twin of `calaman.columnwise_ell1`, the L1
form.

## Shape

Structurally the twin of `calaman.columnwise_ell1` — a host module wrapper
(`interface.cppm`) over a device-compiled translation unit (`columnwise_ell2.cu`),
sharing a launcher bridge (`columnwise_ell2_bridge.h`) across the host/device
line, with an explicit-instantiation unit (`instantiations.cpp`). It takes a
`wwr::wwrStream_t`, not a device handle: it enqueues two kernels and allocates
nothing.

The computation is a per-column reduction followed by a scalar map, so it owns
**no reduction kernel of its own**. Pass one defers to `calaman.reduce_columns`
with `(·)²` as the pre-transform and `+` as the fold, giving the per-column sum
of squares. Pass two takes the elementwise `sqrt` of those sums in place via
`wwr.extension.parallel_for` — the same index-per-thread launch `calaman.gebal`
seeds its `scale[]` with. Both passes run on one stream; in-stream ordering
serialises the sqrt after the reduce. Squaring in the *transform* (not a
fold-in-the-binary-op trick) is what makes a single-row column report `|a|`,
the sqrt of `a²`.

## Scope

Templated over `float` and `double`, matching the rest of calaman. The L2 norm
of a **complex** column is real-valued (`sqrt` of the sum of squared moduli),
i.e. a `T → real` reduction rather than this `T → T` one — a deliberate later
extension, as `calaman.diff_norm` notes for the same reason.
`calaman.reduce_columns` is already generic over that differing value type; only
this wrapper is type-fixed.

## Note on the port

Adapted from a CUDA/Thrust reference that threaded an `int* d_keys` scratch
buffer through the API — that was a `thrust::reduce_by_key` artifact. The
hand-written `reduce_columns` kernel needs no such buffer, so it is gone from the
signature. The reference also factored the sqrt pass into its own `sqrt_each`
static library; here it is a small functor launched inline through
`wwr.extension.parallel_for`, as there is no separate `sqrt_each` module in
calaman.
