// hetrs.cu
//
// The device-kernel half of calaman.hetrs: the pieces the HERMITIAN solve needs
// beyond wrapped BLAS. Everything else in hetrs() is a host composition of
// geru/gemv/swap; this file owns
//
//   * hetrs_scale_row_real: the 1x1 branch's ZDSCAL by the REAL reciprocal of the
//     (real) Hermitian diagonal D(k,k);
//   * hetrs_solve_2x2: the Hermitian 2x2 block solve, with the off-diagonal
//     conjugated on one row (the reference's DCONJG(AKM1K));
//   * hetrs_conj_row: the ZLACGV that brackets each conjugate-transpose gemv.
//
// Complex only (c/z): a real Hermitian matrix is symmetric (calaman.sytrs).
// Arithmetic goes through calaman::device::elem_ops (conj/real_part/div/...), the
// portable spelling. Shared between both backends, like lacpy.cu.
#include "hetrs_bridge.h"

#include "common/elem_ops.cuh"
#include "extension/parallel_for/parallel_for.cuh"

#include <cstddef>

namespace calaman::device {

namespace {

/// @brief Scale one B row by 1/Re(D(k,k)): b[j*ldb] *= 1/real(*akk), per column j
template<typename T>
struct ScaleRowRealFunctor {
  T *const b_;
  const int ldb_;
  const T *const akk_;

  __device__ void operator()(const std::size_t j) const {
    using ops = elem_ops<T>;
    using R = typename ops::real_type;
    const R s = R(1) / ops::real_part(*akk_);
    const std::size_t idx = j * static_cast<std::size_t>(ldb_);
    b_[idx] = ops::scale(b_[idx], s);
  }
};

/// @brief Apply inv of the Hermitian 2x2 D block to two B rows, per column j
///
/// top_div / bot_div are the off-diagonal and its conjugate, assigned per uplo
/// (top_conj picks which row divides by the conjugate). The scaled inputs are read
/// into locals before either row is written, so the overwrite order is immaterial.
template<typename T>
struct Solve2x2Functor {
  T *const b_top_;
  T *const b_bot_;
  const int ldb_;
  const T *const top_diag_;
  const T *const off_;
  const T *const bot_diag_;
  const bool top_conj_;

  __device__ void operator()(const std::size_t j) const {
    using ops = elem_ops<T>;
    using R = typename ops::real_type;
    const T off = *off_;
    const T off_c = ops::conj(off);
    const T top_div = top_conj_ ? off_c : off;
    const T bot_div = top_conj_ ? off : off_c;

    const T akm1 = ops::div(*top_diag_, top_div);
    const T ak = ops::div(*bot_diag_, bot_div);
    const T denom = ops::sub(ops::mul(akm1, ak), ops::from_real(R(1)));

    const std::size_t idx = j * static_cast<std::size_t>(ldb_);
    const T b_top = ops::div(b_top_[idx], top_div);
    const T b_bot = ops::div(b_bot_[idx], bot_div);
    b_top_[idx] = ops::div(ops::sub(ops::mul(ak, b_top), b_bot), denom);
    b_bot_[idx] = ops::div(ops::sub(ops::mul(akm1, b_bot), b_top), denom);
  }
};

/// @brief Conjugate one B row: b[j*ldb] <- conj(b[j*ldb]), per column j
template<typename T>
struct ConjRowFunctor {
  T *const b_;
  const int ldb_;

  __device__ void operator()(const std::size_t j) const {
    const std::size_t idx = j * static_cast<std::size_t>(ldb_);
    b_[idx] = elem_ops<T>::conj(b_[idx]);
  }
};

} // namespace

template<typename T>
void hetrs_scale_row_real(const wwr::wwrStream_t stream, T *const b_row, const int ldb,
                          const int nrhs, const T *const akk) {
  if (nrhs <= 0) {
    return;
  }
  const ScaleRowRealFunctor<T> functor{b_row, ldb, akk};
  wwr::extension::parallel_for<std::size_t>(stream, static_cast<std::size_t>(nrhs), functor);
}

template<typename T>
void hetrs_solve_2x2(const wwr::wwrStream_t stream, T *const b_top, T *const b_bot, const int ldb,
                     const int nrhs, const T *const top_diag, const T *const off,
                     const T *const bot_diag, const bool top_conj) {
  if (nrhs <= 0) {
    return;
  }
  const Solve2x2Functor<T> functor{b_top, b_bot, ldb, top_diag, off, bot_diag, top_conj};
  wwr::extension::parallel_for<std::size_t>(stream, static_cast<std::size_t>(nrhs), functor);
}

template<typename T>
void hetrs_conj_row(const wwr::wwrStream_t stream, T *const b_row, const int ldb, const int nrhs) {
  if (nrhs <= 0) {
    return;
  }
  const ConjRowFunctor<T> functor{b_row, ldb};
  wwr::extension::parallel_for<std::size_t>(stream, static_cast<std::size_t>(nrhs), functor);
}

// One per supported type (complex only), matching hetrs_bridge.h and the module.
template void hetrs_scale_row_real<wwr::wwrFloatComplex>(wwr::wwrStream_t, wwr::wwrFloatComplex *,
                                                         int, int, const wwr::wwrFloatComplex *);
template void hetrs_scale_row_real<wwr::wwrDoubleComplex>(wwr::wwrStream_t, wwr::wwrDoubleComplex *,
                                                          int, int, const wwr::wwrDoubleComplex *);

template void hetrs_solve_2x2<wwr::wwrFloatComplex>(wwr::wwrStream_t, wwr::wwrFloatComplex *,
                                                    wwr::wwrFloatComplex *, int, int,
                                                    const wwr::wwrFloatComplex *,
                                                    const wwr::wwrFloatComplex *,
                                                    const wwr::wwrFloatComplex *, bool);
template void hetrs_solve_2x2<wwr::wwrDoubleComplex>(wwr::wwrStream_t, wwr::wwrDoubleComplex *,
                                                     wwr::wwrDoubleComplex *, int, int,
                                                     const wwr::wwrDoubleComplex *,
                                                     const wwr::wwrDoubleComplex *,
                                                     const wwr::wwrDoubleComplex *, bool);

template void hetrs_conj_row<wwr::wwrFloatComplex>(wwr::wwrStream_t, wwr::wwrFloatComplex *, int,
                                                   int);
template void hetrs_conj_row<wwr::wwrDoubleComplex>(wwr::wwrStream_t, wwr::wwrDoubleComplex *, int,
                                                    int);

} // namespace calaman::device
