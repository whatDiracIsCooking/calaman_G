/**
 * @file interface.cppm
 * @brief Primary interface for calaman.steqr -- all eigenvalues and,
 *        optionally, eigenvectors of a symmetric tridiagonal, LAPACK's ?steqr
 *
 * The implicit QL/QR method on the tridiagonal (d, e): on return @p d holds the
 * eigenvalues in increasing order and, per CompZ, @p z holds the eigenvectors
 * of the tridiagonal (CompZ::I) or Q times them (CompZ::V, Q given in @p z).
 * One single-block kernel (steqr.cu); enqueued on the stream, returns WITHOUT
 * synchronizing, allocates nothing.
 *
 * Mapping from DSTEQR/ZSTEQR (docs/architecture.md §4): COMPZ becomes
 * calaman::CompZ; s/d/c/z become one template over Z's type T, with d, e and
 * WORK in T's real component type (ZSTEQR's real tridiagonal and rotations);
 * WORK becomes a caller byte buffer sized by steqr_bufferSize (the carve-once
 * convention); INFO > 0 (no convergence within 30n iterations) stays a device
 * int, and INFO < 0 becomes an invalid-value Status.
 *
 * Usage:
 *   import calaman.steqr;  // also re-exports CompZ and Status
 *   const std::size_t bytes = calaman::steqr_bufferSize<double>(CompZ::I, n);
 *   calaman::steqr<double>(stream, calaman::CompZ::I, n, d_d, d_e, d_z, ldz,
 *                          d_work, bytes, d_info);
 */

module;

// CLM_TRY / CLM_REQUIRE -- macros, so they arrive by #include in the GMF.
#include "error_handling/error_macros.h"

#include "steqr_bridge.h"

export module calaman.steqr;

import std;
import wwr.runtime_api;
import wwr.complex;    // wwrFloatComplex, wwrDoubleComplex
import calaman.common; // CompZ (:enums), ComplexToRealType, carve_workspace

// export import: steqr RETURNS calaman::Status, so a consumer must see its
// members, not just its name.
export import calaman.error_handling;

namespace calaman {

// Re-exported so `import calaman.steqr;` alone names the selector.
export using calaman::CompZ;

namespace steqr_detail {

/// @brief steqr()'s workspace: the 2(n-1) saved rotations (cosines, then
///        sines) -- only when vectors are wanted, as DSTEQR's WORK; real
///        (@p R) for a complex Z too, as ZSTEQR's
template<typename R>
struct SteqrSlices {
  R *rot = nullptr;

  void carve(WorkspaceLayout &layout, const CompZ compz, const int n) {
    if (compz != CompZ::N && n > 1) {
      rot = layout.fixed<R>(2 * static_cast<std::size_t>(n - 1));
    }
  }
};

static_assert(slices_for<SteqrSlices<double>, CompZ, int>);

} // namespace steqr_detail

/// @brief Device workspace steqr() needs, in bytes (0 for CompZ::N or n <= 1)
export template<typename T>
std::size_t steqr_bufferSize(const CompZ compz, const int n) {
  using Slices = steqr_detail::SteqrSlices<ComplexToRealType<T>>;
  return carve_workspace<Slices>(nullptr, nullptr, compz, n);
}

/// @brief Eigen-decompose the order-@p n symmetric tridiagonal (d, e) on @p stream
///
/// @p d (n) is overwritten by the ascending eigenvalues, @p e (n-1) is
/// destroyed; @p z (n-by-n, leading dim @p ldz) is unreferenced for CompZ::N.
/// @p info (device int) gets 0, or the number of off-diagonals that failed to
/// converge (d then holds the eigenvalues found so far, unsorted).
///
/// @tparam T Type of Z; float, double, wwrFloatComplex or wwrDoubleComplex
/// @param d, e The tridiagonal, real (ComplexToRealType<T>) for every T
/// @param work_bytes Size of @p d_work; at least steqr_bufferSize(compz, n)
/// @return Success, the launch error, or invalid-value for a bad argument
export template<typename T>
Status steqr(const wwr::wwrStream_t stream, const CompZ compz, const int n,
             ComplexToRealType<T> *const d, ComplexToRealType<T> *const e, T *const z,
             const int ldz, void *const d_work, const std::size_t work_bytes, int *const info) {
  using R = ComplexToRealType<T>;
  const bool wantz = compz != CompZ::N;
  CLM_REQUIRE(n >= 0, wwr::wwrErrorInvalidValue);
  CLM_REQUIRE(ldz >= 1 && (!wantz || ldz >= std::max(1, n)), wwr::wwrErrorInvalidValue);
  CLM_REQUIRE(info != nullptr, wwr::wwrErrorInvalidValue);
  CLM_REQUIRE(n == 0 || d != nullptr, wwr::wwrErrorInvalidValue);
  CLM_REQUIRE(n <= 1 || e != nullptr, wwr::wwrErrorInvalidValue);
  CLM_REQUIRE(!wantz || n == 0 || z != nullptr, wwr::wwrErrorInvalidValue);

  steqr_detail::SteqrSlices<R> ws;
  const std::size_t need = carve_workspace(d_work, &ws, compz, n);
  CLM_REQUIRE(work_bytes >= need && (need == 0 || d_work != nullptr), wwr::wwrErrorInvalidValue);

  device::steqr<T, R>(stream, compz, n, d, e, z, ldz, ws.rot, info);
  CLM_TRY(wwr::wwrGetLastError());
  return wwr::wwrSuccess;
}

extern template std::size_t steqr_bufferSize<float>(CompZ, int);
extern template std::size_t steqr_bufferSize<double>(CompZ, int);
extern template std::size_t steqr_bufferSize<wwr::wwrFloatComplex>(CompZ, int);
extern template std::size_t steqr_bufferSize<wwr::wwrDoubleComplex>(CompZ, int);
extern template Status steqr<float>(wwr::wwrStream_t, CompZ, int, float *, float *, float *, int,
                                    void *, std::size_t, int *);
extern template Status steqr<double>(wwr::wwrStream_t, CompZ, int, double *, double *, double *,
                                     int, void *, std::size_t, int *);
extern template Status steqr<wwr::wwrFloatComplex>(wwr::wwrStream_t, CompZ, int, float *, float *,
                                                   wwr::wwrFloatComplex *, int, void *,
                                                   std::size_t, int *);
extern template Status steqr<wwr::wwrDoubleComplex>(wwr::wwrStream_t, CompZ, int, double *,
                                                    double *, wwr::wwrDoubleComplex *, int,
                                                    void *, std::size_t, int *);

} // namespace calaman
