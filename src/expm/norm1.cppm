/**
 * @file norm1.cppm
 * @brief The induced matrix 1-norm -- the single measurement the whole plan
 *        hangs off
 *
 * The :norm1 partition of calaman.expm. A thin host wrapper over the two fused
 * device launchers (abs_colsums, max_reduce, in expm_bridge.h); exported on its
 * own because callers outside the module (calaman.cg_unitary) take the 1-norm
 * without wanting the exponential.
 */

module;

#include "expm_bridge.h"

export module calaman.expm:norm1;

import wwr.runtime_api;     // wwrStream_t, wwrMemcpyAsync, wwrStreamSynchronize, wwrMemcpyDeviceToHost
import wwr.complex;         // wwrFloatComplex, wwrDoubleComplex (the extern-template list)
import calaman.common;  // usual_fp, ComplexToRealType

namespace calaman {

// Not an `export namespace` block: an explicit instantiation declaration
// (`extern template`) cannot be exported, so the template carries its own
// `export` and the declarations below sit in the plain namespace.

/**
 * @brief Maximum absolute column sum of an n x n column-major matrix.
 *
 * Complex elements contribute their TRUE modulus, so this is the induced
 * 1-norm, not an |Re| + |Im| surrogate. The per-column sums land in @p d_colsum
 * and are then reduced in place.
 *
 * @tparam T       Element type (one of the four usual_fp types)
 * @param stream   Stream; work is enqueued here.
 * @param n        Matrix dimension.
 * @param d_A      Device matrix, column-major, leading dimension @p lda.
 * @param lda      Leading dimension of @p d_A (>= n).
 * @param d_colsum Device scratch, n reals (contents discarded).
 * @return The 1-norm, on the host. NaN if any element is NaN.
 *
 * @warning Synchronizes @p stream in order to return a host value.
 */
export template<calaman::usual_fp T>
calaman::ComplexToRealType<T> matrix_norm1(wwr::wwrStream_t stream, const int n, const T *d_A,
                                       const int lda, calaman::ComplexToRealType<T> *d_colsum) {
  using R = calaman::ComplexToRealType<T>;
  device::abs_colsums<T, R>(stream, n, d_A, lda, d_colsum);
  device::max_reduce<R>(stream, n, d_colsum);

  R host{};
  wwr::wwrMemcpyAsync(&host, d_colsum, sizeof(R), wwr::wwrMemcpyDeviceToHost, stream);
  wwr::wwrStreamSynchronize(stream);
  return host;
}

// Instantiated once in instantiations.cpp -- its body names the .cu-side
// launchers declared only in the global module fragment, so an importer never
// re-instantiates it.
extern template calaman::ComplexToRealType<float> matrix_norm1<float>(wwr::wwrStream_t, int,
                                                                  const float *, int, float *);
extern template calaman::ComplexToRealType<double> matrix_norm1<double>(wwr::wwrStream_t, int,
                                                                    const double *, int, double *);
extern template calaman::ComplexToRealType<wwr::wwrFloatComplex>
matrix_norm1<wwr::wwrFloatComplex>(wwr::wwrStream_t, int, const wwr::wwrFloatComplex *, int,
                                   float *);
extern template calaman::ComplexToRealType<wwr::wwrDoubleComplex>
matrix_norm1<wwr::wwrDoubleComplex>(wwr::wwrStream_t, int, const wwr::wwrDoubleComplex *, int,
                                    double *);

} // namespace calaman
