/**
 * @file interface.cppm
 * @brief Primary interface for calaman.lacp2 -- copy a real matrix into a
 *        complex one, LAPACK's ?lacp2
 *
 * Copies all, the upper, or the lower triangle of the m-by-n real column-major
 * matrix A into the complex B, with imaginary part 0. calaman.lacpy's surface:
 * a Region, a bare wwrStream_t, nothing allocated, and B outside the copied
 * region (including its ldb padding rows) untouched. Enqueued on the stream;
 * returns WITHOUT synchronizing. ?lacp2 reports no INFO.
 *
 * Mapping from ZLACP2 (docs/architecture.md §4):
 *
 * | ZLACP2           | calaman::lacp2                                 |
 * |------------------|------------------------------------------------|
 * | CHARACTER UPLO   | calaman::Region enum                           |
 * | c/z variants     | one template over ComplexT (complex_fp)        |
 * | DOUBLE A         | ComplexToRealType<ComplexT>                    |
 * | INTEGER extents  | std::size_t; LDA / LDB kept                    |
 *
 * Usage:
 *   import calaman.lacp2;     // also re-exports calaman::Region and Status
 *   // d_a: m x n real (lda); d_b: m x n complex (ldb), both device
 *   calaman::lacp2<wwr::wwrDoubleComplex>(stream, calaman::Region::A, m, n,
 *                                         d_a, lda, d_b, ldb);
 */

module;

// CLM_TRY arrives by #include in the GMF; lacp2_bridge.h declares the .cu
// launcher there too.
#include "error_handling/error_macros.h"

#include "lacp2_bridge.h"

export module calaman.lacp2;

import std;
import wwr.runtime_api; // wwrStream_t, wwrGetLastError, wwrSuccess
import wwr.complex;     // wwrFloatComplex, wwrDoubleComplex (extern template list)
import calaman.common;  // Region, complex_fp, ComplexToRealType

// export import: lacp2 RETURNS calaman::Status, so a consumer must see its
// members, not just its name.
export import calaman.error_handling;

namespace calaman {

// Re-exported so `import calaman.lacp2;` alone names calaman::Region.
export using calaman::Region;

/// @brief B <- (A, 0) over @p region of the m-by-n real A, on @p stream (?lacp2)
///
/// Enqueues the copy and returns without synchronizing; B outside @p region is
/// left untouched. Enqueues nothing when @p m or @p n is 0.
///
/// @tparam ComplexT Complex element type (wwrFloatComplex, wwrDoubleComplex)
/// @param stream Stream the copy is enqueued on; A and B live on its device
/// @param region Which part of A to copy (Region::U, Region::L, Region::A)
/// @param d_a Real source, column-major, leading dimension @p lda >= m
/// @param d_b Complex destination, column-major, leading dimension @p ldb >= m
/// @return Success, or the runtime error the kernel launch reported
export template<calaman::complex_fp ComplexT>
Status lacp2(const wwr::wwrStream_t stream, const Region region, const std::size_t m,
             const std::size_t n, const ComplexToRealType<ComplexT> *d_a, const std::size_t lda,
             ComplexT *d_b, const std::size_t ldb) {
  if (m == 0 || n == 0) {
    return wwr::wwrSuccess;
  }
  device::lacp2(stream, region, m, n, d_a, lda, d_b, ldb);
  CLM_TRY(wwr::wwrGetLastError());
  return wwr::wwrSuccess;
}

extern template Status lacp2<wwr::wwrFloatComplex>(wwr::wwrStream_t, Region, std::size_t,
                                                   std::size_t, const float *, std::size_t,
                                                   wwr::wwrFloatComplex *, std::size_t);
extern template Status lacp2<wwr::wwrDoubleComplex>(wwr::wwrStream_t, Region, std::size_t,
                                                    std::size_t, const double *, std::size_t,
                                                    wwr::wwrDoubleComplex *, std::size_t);

} // namespace calaman
