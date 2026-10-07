/**
 * @file linear_operator.cppm
 * @brief The calaman.linear_operator module -- linear_operator, the block
 *        operator every matrix-free solver constrains on
 *
 * A model owns its operator's data and applies it to a block of device
 * vectors, so a solver written against the concept never reads A's entries.
 * One concept for feast, lanczos and davidson (docs/architecture.md §9); a
 * single-vector callback is the k = 1 case.
 *
 * Not `calaman.operator`: `operator` is a keyword, so it cannot be a
 * module-name component.
 *
 * Imports only std, wwr.runtime_api (the stream) and calaman.error_handling
 * (Status): no device code.
 *
 * Usage:
 *   import calaman.linear_operator;
 *
 *   template<calaman::linear_operator<double> Op>
 *   calaman::Status step(Op &op, wwr::wwrStream_t s, const double *X, double *Y) {
 *     return op.apply(s, 4, X, Y); // Y = A X, n x 4
 *   }
 */

export module calaman.linear_operator;

import std;
import wwr.runtime_api;               // wwrStream_t
export import calaman.error_handling; // Status -- the cross-domain return type

export namespace calaman {

/// @brief op.apply(stream, k, X, Y) enqueues Y = A X on @p stream.
///
/// X and Y are n x k, device-resident, column-major with leading dimension n,
/// and do not alias; n is the model's own. A model may hold state, so apply is
/// called on a non-const @p op.
template<class Op, class T>
concept linear_operator = requires(Op &op, wwr::wwrStream_t stream, int k, const T *X, T *Y) {
  { op.apply(stream, k, X, Y) } -> std::convertible_to<Status>;
};

} // namespace calaman

namespace calaman::detail {

/// The minimal model: apply and nothing else. Declared only, never called.
struct MinimalOperator {
  Status apply(wwr::wwrStream_t stream, int k, const double *X, double *Y);
};
static_assert(linear_operator<MinimalOperator, double>);
static_assert(!linear_operator<MinimalOperator, float>);
static_assert(!linear_operator<int, double>);

} // namespace calaman::detail
