# calaman.iterative

The outcome report every iterative method in `src/` shares: `IterationInfo`,
the `stop_reason` concept, and `converged()`. Pure host, `std` only.

```cpp
template <class E>
concept stop_reason = std::is_enum_v<E> && requires {
  E::Converged; E::MaxIterations; E::NumericalFailure;
};

template <stop_reason Reason>
struct IterationInfo {
  int iterations = 0;                     // outer iterations, in the method's own unit
  Reason reason = Reason::MaxIterations;
};

template <stop_reason Reason>
constexpr bool converged(const IterationInfo<Reason> &i) noexcept;
```

## The rules

- **Non-convergence is an outcome, not an error.** An iterative routine returns
  success whenever it stops on one of its own conditions, `MaxIterations`
  included; the caller reads `converged(info)` or `info.reason`. A non-success
  `Status` means a fault: a bad argument, or a BLAS/solver/runtime call that
  failed.
- **Each method keeps its own stop-reason enum.** It must hold the shared three
  (`static_assert(stop_reason<…>)` next to it) and may add its own. There is no
  shared union enum.
- **Only fields that mean the same thing in every method go in the base.** The
  final convergence measure stays per-method — cg's `gradient_norm_sq`, feast's
  `max_residual` and nnls's `residual_norm` bound different quantities.
- **Options stay per-method**, since `tol` bounds a different quantity in each.
  The iteration budget is spelled `max_iterations` everywhere, to match
  `IterationInfo::iterations`.

## Adopters

| Module | Info | Extra stop reasons |
|---|---|---|
| `calaman.cg_unitary` | `CgInfo : IterationInfo<CgStopReason>` | `LineSearchFailed` |
| `calaman.feast` | `FeastInfo : IterationInfo<FeastStopReason>` | `SubspaceTooSmall` |
| `calaman.nnls` | `NnlsInfo : IterationInfo<NnlsStopReason>` | — |

Each re-exports `calaman.iterative`, so importing the method is enough to call
`converged(info)`. `davidson` and `lanczos` have not migrated yet: they still
report through a `Result` with a `bool converged`.

Tests assert convergence with `EXPECT_CONVERGED(info)` from
`test/shared/expect_converged.h`, which prints `reason` and `iterations` on
failure.
