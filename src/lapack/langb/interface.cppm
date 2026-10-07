/**
 * @file interface.cppm
 * @brief Primary interface for calaman.langb -- the norm of a general band
 *        matrix, LAPACK's ?langb
 *
 * Returns the largest absolute entry, 1-norm, infinity-norm or Frobenius norm
 * of the n-by-n band matrix A with kl sub- and ku super-diagonals, held in
 * LAPACK band storage: A(i,j) at AB(ku+i-j, j). Only in-matrix band entries are
 * read, so the unused corners of AB and its padding may hold anything. The
 * result is written to a device scalar and the call returns WITHOUT
 * synchronizing; n == 0 writes 0, as DLANGB returns 0. Like calaman.lansy, it
 * allocates an n-element scratch on @p stream (stream-ordered).
 *
 * Mapping from DLANGB (docs/architecture.md §4): NORM becomes MatrixNorm; s/d/c/z
 * become one template over T, whose norm is always real (ComplexToRealType<T>).
 * Divergence: the Frobenius norm is a plain sum of squares, not DLANGB's scaled
 * ?lassq -- the calaman.lange precedent.
 *
 * Usage:
 *   import calaman.langb;  // also re-exports MatrixNorm and Status
 *   // d_AB: (kl+ku+1)-by-n (or taller) device band storage, column-major
 *   calaman::langb<double>(stream, calaman::MatrixNorm::one, n, kl, ku, d_AB, ldab,
 *                          d_result);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment.
#include "error_handling/error_macros.h"

#include "langb_bridge.h"

export module calaman.langb;

import std;
import wwr.runtime_api; // wwrStream_t, wwrMallocAsync/FreeAsync/MemsetAsync, wwrGetLastError
import wwr.complex;     // wwrFloatComplex, wwrDoubleComplex (extern template list)
import calaman.common;  // MatrixNorm (:enums), ComplexToRealType

// export import: langb RETURNS calaman::Status, so a consumer must see its
// members, not just its name -- the re-export lange does.
export import calaman.error_handling;

namespace calaman {

// Re-exported so `import calaman.langb;` alone names the selector.
export using calaman::MatrixNorm;

/// @brief Norm of the n-by-n band matrix A held in band storage @p d_AB (?langb)
///
/// Writes the @p which norm to @p d_result and returns without synchronizing;
/// writes 0 when @p n is 0. The return carries an allocation or launch failure.
///
/// @tparam T Element type; float, double, wwrFloatComplex or wwrDoubleComplex
/// @param stream   Stream the work is enqueued on; all pointers live on its device
/// @param which    Which norm to compute (max_abs, one, inf, frobenius)
/// @param n        Order of A
/// @param kl       Number of sub-diagonals of A
/// @param ku       Number of super-diagonals of A
/// @param d_AB     Device band storage, column-major; A(i,j) at d_AB[ku+i-j + j*ldab]
/// @param ldab     Leading dimension of @p d_AB; ldab >= kl+ku+1
/// @param d_result Device scalar receiving the (real) norm
/// @return Success, or the first failing step's error (allocation or launch)
export template<typename T>
Status langb(const wwr::wwrStream_t stream, const MatrixNorm which, const std::size_t n,
             const std::size_t kl, const std::size_t ku, const T *const d_AB,
             const std::size_t ldab, ComplexToRealType<T> *const d_result) {
  using R = ComplexToRealType<T>;
  if (n == 0) {
    return wwr::wwrMemsetAsync(d_result, 0, sizeof(R), stream);
  }
  R *d_scratch = nullptr;
  CLM_TRY(wwr::wwrMallocAsync(reinterpret_cast<void **>(&d_scratch), n * sizeof(R), stream));

  device::langb<T, R>(stream, which, n, kl, ku, d_AB, ldab, d_result, d_scratch);
  // Capture the sticky launch error BEFORE the free so a free failure cannot
  // mask it; the stream-ordered free's own failure is discarded, as in lange.
  const wwr::wwrError_t launch_status = wwr::wwrGetLastError();
  static_cast<void>(wwr::wwrFreeAsync(d_scratch, stream));
  return launch_status;
}

extern template Status langb<float>(wwr::wwrStream_t, MatrixNorm, std::size_t, std::size_t,
                                    std::size_t, const float *, std::size_t, float *);
extern template Status langb<double>(wwr::wwrStream_t, MatrixNorm, std::size_t, std::size_t,
                                     std::size_t, const double *, std::size_t, double *);
extern template Status langb<wwr::wwrFloatComplex>(wwr::wwrStream_t, MatrixNorm, std::size_t,
                                                   std::size_t, std::size_t,
                                                   const wwr::wwrFloatComplex *, std::size_t,
                                                   float *);
extern template Status langb<wwr::wwrDoubleComplex>(wwr::wwrStream_t, MatrixNorm, std::size_t,
                                                    std::size_t, std::size_t,
                                                    const wwr::wwrDoubleComplex *, std::size_t,
                                                    double *);

} // namespace calaman
