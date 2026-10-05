/**
 * @file hetf2.cppm
 * @brief The calaman.hetf2 module -- the unblocked Hermitian Bunch-Kaufman
 *        factorization of a Hermitian indefinite matrix, LAPACK's ?hetf2
 *
 * Factors A = U D U^H (Uplo::U) or L D L^H (Uplo::L) in place, with D block
 * diagonal (real 1x1 and Hermitian 2x2 blocks) and the symmetric interchanges
 * recorded in @p d_ipiv (1-based, negative pairs for a 2x2 block). COMPLEX ONLY
 * (c/z): a real Hermitian matrix is symmetric, which the vendor ?sytrf covers
 * (calaman.sysv). This is the from-scratch factorization the vendors omit for
 * Hermitian; calaman.hesv pairs it with calaman.hetrs for the full solve.
 *
 * Structure mirrors the reference ?hetf2: a host loop walks the columns, runs the
 * Bunch-Kaufman pivot search (wwr::iamax plus a few per-step scalar reads) and
 * decisions host-side, then drives the matrix work on the handle's stream -- the
 * Hermitian interchange and the 2x2 rank-2 update as device kernels (hetf2.cu),
 * the 1x1 rank-1 update as wwr::her + wwr::scal. The pivots and info are
 * accumulated host-side and uploaded once at the end.
 *
 * Allocation-free shipped surface (CLAUDE.md): A, @p d_ipiv, @p d_info and
 * @p d_work are caller-provided device pointers; @p d_work is >= 2n elements (the
 * two factor-column scratch vectors a 2x2 update needs). Requires the handle's
 * DEFAULT (host) pointer mode -- iamax and the BLAS scalars are host-side.
 */

module;

#include "hetf2_bridge.h"
// CLM_TRY -- a macro, so it arrives by #include in the global module fragment.
#include "error_handling/error_macros.h"

export module calaman.hetf2;

import std;               // std::vector, std::complex, std::abs, std::isnan, std::sqrt
import wwr.blas;          // wwrblasHandle_t, WWRBLAS_FILL_MODE_*, wwrblasGetStream
import wwr.runtime_api;   // wwrMemcpy(Async), wwrStreamSynchronize
import wwr.complex;       // wwrFloatComplex, wwrDoubleComplex
import wwr.wrappers.blas; // iamax, her, scal
import calaman.common;    // Uplo, complex_fp, ComplexToRealType

export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

namespace hetf2_detail {

using std::size_t;

/// @brief Read one device element to the host as a std::complex (layout-identical)
template<typename T, typename R>
Status read1(wwr::wwrStream_t stream, const T *p, std::complex<R> *out) {
  CLM_TRY(wwr::wwrMemcpyAsync(out, p, sizeof(T), wwr::wwrMemcpyDeviceToHost, stream));
  CLM_TRY(wwr::wwrStreamSynchronize(stream));
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/// @brief The |Re| + |Im| magnitude the Bunch-Kaufman search uses (LAPACK CABS1)
template<typename R>
R cabs1(const std::complex<R> &z) {
  return std::abs(z.real()) + std::abs(z.imag());
}

/// @brief Reinterpret a host std::complex as the layout-identical device element
template<typename T, typename R>
T to_elem(const std::complex<R> &z) {
  T out;
  std::memcpy(&out, &z, sizeof(T));
  return out;
}

} // namespace hetf2_detail

/// @brief Device workspace, in elements of T, required by hetf2() (= 2n)
export template<complex_fp T>
Status hetf2_bufferSize(const int n, int *lwork) {
  if (n < 0) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }
  *lwork = 2 * n;
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/// @brief Unblocked Hermitian Bunch-Kaufman factorization (LAPACK ?hetf2)
///
/// Overwrites the @p uplo triangle of A with the factor, records the pivots in
/// @p d_ipiv, and writes the first zero/NaN pivot column (1-based, else 0) to
/// @p d_info. Complex only. The handle must be in host pointer mode and bound to
/// the stream all work is enqueued on.
///
/// @tparam T Complex element type (wwrFloatComplex or wwrDoubleComplex)
/// @param handle GPU BLAS handle in host pointer mode; all arrays live on its device
/// @param uplo Which triangle of A is referenced and overwritten with the factor
/// @param n Order of the Hermitian matrix A
/// @param d_A Device matrix, n by n, column-major; its @p uplo triangle is factored in place
/// @param lda Leading dimension of A (>= max(1, n))
/// @param d_ipiv Device int array, length n; filled with the 1-based pivot sequence
/// @param d_info Device int: 0 on success, else the 1-based first zero/NaN pivot column
/// @param d_work Device workspace, >= 2n elements of T
/// @return The Status of the failing step, otherwise WWRBLAS_STATUS_SUCCESS
export template<complex_fp T>
Status hetf2(wwr::wwrblasHandle_t handle, const Uplo uplo, const int n, T *d_A, const int lda,
             int *d_ipiv, int *d_info, T *d_work) {
  using R = ComplexToRealType<T>;
  using C = std::complex<R>;
  using namespace hetf2_detail;

  if (n < 0 || lda < std::max(1, n)) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }
  int info_host = 0;
  if (n == 0) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  wwr::wwrStream_t stream{};
  CLM_TRY(wwr::wwrblasGetStream(handle, &stream));

  const bool upper = uplo == Uplo::U;
  const R alpha = (R{1} + std::sqrt(R{17})) / R{8};
  std::vector<int> piv(static_cast<std::size_t>(n), 0);
  T *const wa = d_work;
  T *const wb = d_work + n;
  C tmp;

  auto readv = [&](const T *p) -> C {
    read1<T, R>(stream, p, &tmp);
    return tmp;
  };

  int k = upper ? n - 1 : 0;
  Status st = wwr::WWRBLAS_STATUS_SUCCESS;
  while (upper ? (k >= 0) : (k < n)) {
    int kstep = 1;
    const C akk = readv(d_A + static_cast<size_t>(k) * lda + k);
    const R absakk = std::abs(akk.real());

    int imax = -1;
    R colmax = R{0};
    if (upper ? (k > 0) : (k < n - 1)) {
      int pos = 0;
      if (upper) {
        CLM_TRY(wwr::iamax<T, int>(handle, k, d_A + static_cast<size_t>(k) * lda, 1, &pos));
        imax = pos - 1;
      } else {
        CLM_TRY(wwr::iamax<T, int>(handle, n - k - 1, d_A + static_cast<size_t>(k) * lda + (k + 1),
                                   1, &pos));
        imax = k + pos;
      }
      colmax = cabs1(readv(d_A + static_cast<size_t>(k) * lda + imax));
    }

    int kp = k;
    const bool singular = (std::max(absakk, colmax) == R{0}) || std::isnan(absakk);
    if (singular) {
      if (info_host == 0) {
        info_host = k + 1;
      }
      kp = k;
    } else if (absakk >= alpha * colmax) {
      kp = k;
    } else {
      // ROWMAX over row imax.
      int pos = 0;
      R rowmax = R{0};
      if (upper) {
        CLM_TRY(wwr::iamax<T, int>(handle, k - imax,
                                   d_A + static_cast<size_t>(imax + 1) * lda + imax, lda, &pos));
        const int jmax = imax + pos;
        rowmax = cabs1(readv(d_A + static_cast<size_t>(jmax) * lda + imax));
        if (imax > 0) {
          CLM_TRY(wwr::iamax<T, int>(handle, imax, d_A + static_cast<size_t>(imax) * lda, 1, &pos));
          const int jmax2 = pos - 1;
          rowmax = std::max(rowmax, cabs1(readv(d_A + static_cast<size_t>(imax) * lda + jmax2)));
        }
      } else {
        CLM_TRY(wwr::iamax<T, int>(handle, imax - k, d_A + static_cast<size_t>(k) * lda + imax, lda,
                                   &pos));
        const int jmax = k + pos - 1;
        rowmax = cabs1(readv(d_A + static_cast<size_t>(jmax) * lda + imax));
        if (imax < n - 1) {
          CLM_TRY(wwr::iamax<T, int>(handle, n - 1 - imax,
                                     d_A + static_cast<size_t>(imax) * lda + (imax + 1), 1, &pos));
          const int jmax2 = imax + pos;
          rowmax = std::max(rowmax, cabs1(readv(d_A + static_cast<size_t>(imax) * lda + jmax2)));
        }
      }
      const R aimax = std::abs(readv(d_A + static_cast<size_t>(imax) * lda + imax).real());
      if (absakk >= alpha * colmax * (colmax / rowmax)) {
        kp = k;
      } else if (aimax >= alpha * rowmax) {
        kp = imax;
      } else {
        kp = imax;
        kstep = 2;
      }
    }

    const int kk = upper ? (k - kstep + 1) : (k + kstep - 1);
    device::hetf2_interchange<T>(stream, upper, n, d_A, static_cast<size_t>(lda), k, kk, kp, kstep);

    if (!singular) {
      if (kstep == 1) {
        const C akk2 = readv(d_A + static_cast<size_t>(k) * lda + k);
        const R r1 = R{1} / akk2.real();
        const R neg_r1 = -r1;
        const T scal_a = to_elem<T, R>(C(r1, R{0}));
        if (upper && k > 0) {
          CLM_TRY(wwr::her<T, int>(handle, wwr::WWRBLAS_FILL_MODE_UPPER, k, &neg_r1,
                                   d_A + static_cast<size_t>(k) * lda, 1, d_A, lda));
          CLM_TRY(wwr::scal<T, int>(handle, k, &scal_a, d_A + static_cast<size_t>(k) * lda, 1));
        } else if (!upper && k < n - 1) {
          CLM_TRY(wwr::her<T, int>(handle, wwr::WWRBLAS_FILL_MODE_LOWER, n - k - 1, &neg_r1,
                                   d_A + static_cast<size_t>(k) * lda + (k + 1), 1,
                                   d_A + static_cast<size_t>(k + 1) * lda + (k + 1), lda));
          CLM_TRY(wwr::scal<T, int>(handle, n - k - 1, &scal_a,
                                    d_A + static_cast<size_t>(k) * lda + (k + 1), 1));
        }
      } else if (upper ? (k > 1) : (k < n - 2)) {
        // 2x2 pivot: inv-D scalars from the (interchanged) block, then rank-2.
        const int ktop = upper ? k - 1 : k;
        const int kbot = upper ? k : k + 1;
        const C diag_top = readv(d_A + static_cast<size_t>(ktop) * lda + ktop);
        const C diag_bot = readv(d_A + static_cast<size_t>(kbot) * lda + kbot);
        // The 2x2 off-diagonal lives in the stored triangle: A(k-1,k) upper
        // (super-diagonal), A(k+1,k) lower (sub-diagonal).
        const int off_row = upper ? ktop : kbot;
        const int off_col = upper ? kbot : ktop;
        const C off = readv(d_A + static_cast<size_t>(off_col) * lda + off_row);
        const R dnorm = std::abs(off);                                     // DLAPY2
        // d_a pairs with column k's W, d_b with the partner column's W.
        const R d11 = diag_bot.real() / dnorm;
        const R d22 = diag_top.real() / dnorm;
        const R tt = R{1} / (d11 * d22 - R{1});
        const R dfac = tt / dnorm;
        const double d_a = upper ? static_cast<double>(d22) : static_cast<double>(d11);
        const double d_b = upper ? static_cast<double>(d11) : static_cast<double>(d22);
        const C doff_c(off.real() / dnorm, off.imag() / dnorm);
        device::hetf2_rank2<T>(stream, upper, n, d_A, static_cast<size_t>(lda), k, d_a, d_b,
                               to_elem<T, R>(doff_c), static_cast<double>(dfac), wa, wb);
      }
    }

    if (kstep == 1) {
      piv[static_cast<std::size_t>(k)] = kp + 1;
    } else {
      piv[static_cast<std::size_t>(k)] = -(kp + 1);
      piv[static_cast<std::size_t>(upper ? k - 1 : k + 1)] = -(kp + 1);
    }

    k = upper ? (k - kstep) : (k + kstep);
  }

  // Publish the pivots and info to the device (one transfer each).
  CLM_TRY(wwr::wwrMemcpyAsync(d_ipiv, piv.data(), static_cast<std::size_t>(n) * sizeof(int),
                              wwr::wwrMemcpyHostToDevice, stream));
  CLM_TRY(wwr::wwrMemcpyAsync(d_info, &info_host, sizeof(int), wwr::wwrMemcpyHostToDevice, stream));
  CLM_TRY(wwr::wwrStreamSynchronize(stream));
  return st;
}

} // namespace calaman
