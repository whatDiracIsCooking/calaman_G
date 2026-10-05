// sytrs.cu
//
// The device-kernel half of calaman.sytrs: applying the block-diagonal D^-1 of a
// Bunch-Kaufman factorization to the right-hand sides. Everything else in
// sytrs() is a host composition of wrapped BLAS (ger/geru/gemv/swap); this file
// owns only the two D-block applies, the pieces no BLAS call expresses:
//
//   * sytrs_scale_row: the 1x1 branch's DSCAL(1/A(K,K)) over one B row;
//   * sytrs_solve_2x2: the 2x2 branch's symmetric block solve over two B rows.
//
// Both ride wwr.extension.parallel_for (one thread per right-hand-side column)
// and read the stored factor scalars straight off the device, so the whole
// solve stays on the handle's stream with no host round-trip per pivot. Complex
// arithmetic goes through calaman::device::elem_ops, the portable spelling, so
// the symmetric (unconjugated) forms cover all four element types unchanged.
// Shared between both backends, like lacpy.cu; the .cu extension is all CMake
// needs (under HIP the CMakeLists forces -x hip), so no __CUDACC__.
#include "sytrs_bridge.h"

#include "common/elem_ops.cuh"
#include <extension/parallel_for/parallel_for.cuh>

#include <cstddef>

namespace calaman::device {

namespace {

/// @brief Scale one B row by 1/D(k,k): b[j*ldb] <- b[j*ldb] / *akk, per column j
template<typename T>
struct ScaleRowFunctor {
  T *const b_;
  const int ldb_;
  const T *const akk_;

  __device__ void operator()(const std::size_t j) const {
    const std::size_t idx = j * static_cast<std::size_t>(ldb_);
    b_[idx] = elem_ops<T>::div(b_[idx], *akk_);
  }
};

/// @brief Apply inv of the symmetric 2x2 D block to two B rows, per column j
///
/// With off = *off_, akm1 = *top_diag_/off, ak = *bot_diag_/off and
/// denom = akm1*ak - 1, each column sets
///   b_top <- (ak*b_top/off - b_bot/off)/denom
///   b_bot <- (akm1*b_bot/off - b_top/off)/denom
/// the symmetric-2x2 inverse the reference ?sytrs spells as BKM1/BK/DENOM. The
/// scaled inputs are read into locals before either row is written, so the two
/// rows may overlap nothing and the overwrite order does not matter.
template<typename T>
struct Solve2x2Functor {
  T *const b_top_;
  T *const b_bot_;
  const int ldb_;
  const T *const top_diag_;
  const T *const off_;
  const T *const bot_diag_;

  __device__ void operator()(const std::size_t j) const {
    using ops = elem_ops<T>;
    const T off = *off_;
    const T akm1 = ops::div(*top_diag_, off);
    const T ak = ops::div(*bot_diag_, off);
    const T denom = ops::sub(ops::mul(akm1, ak), ops::from_real(typename ops::real_type(1)));

    const std::size_t idx = j * static_cast<std::size_t>(ldb_);
    const T b_top = ops::div(b_top_[idx], off);
    const T b_bot = ops::div(b_bot_[idx], off);
    b_top_[idx] = ops::div(ops::sub(ops::mul(ak, b_top), b_bot), denom);
    b_bot_[idx] = ops::div(ops::sub(ops::mul(akm1, b_bot), b_top), denom);
  }
};

} // namespace

template<typename T>
void sytrs_scale_row(const wwr::wwrStream_t stream, T *const b_row, const int ldb, const int nrhs,
                     const T *const akk) {
  if (nrhs <= 0) {
    return;
  }
  const ScaleRowFunctor<T> functor{b_row, ldb, akk};
  wwr::extension::parallel_for<std::size_t>(stream, static_cast<std::size_t>(nrhs), functor);
}

template<typename T>
void sytrs_solve_2x2(const wwr::wwrStream_t stream, T *const b_top, T *const b_bot, const int ldb,
                     const int nrhs, const T *const top_diag, const T *const off,
                     const T *const bot_diag) {
  if (nrhs <= 0) {
    return;
  }
  const Solve2x2Functor<T> functor{b_top, b_bot, ldb, top_diag, off, bot_diag};
  wwr::extension::parallel_for<std::size_t>(stream, static_cast<std::size_t>(nrhs), functor);
}

// One per supported type, matching sytrs_bridge.h and the module's use sites.
template void sytrs_scale_row<float>(wwr::wwrStream_t, float *, int, int, const float *);
template void sytrs_scale_row<double>(wwr::wwrStream_t, double *, int, int, const double *);
template void sytrs_scale_row<wwr::wwrFloatComplex>(wwr::wwrStream_t, wwr::wwrFloatComplex *, int,
                                                    int, const wwr::wwrFloatComplex *);
template void sytrs_scale_row<wwr::wwrDoubleComplex>(wwr::wwrStream_t, wwr::wwrDoubleComplex *, int,
                                                     int, const wwr::wwrDoubleComplex *);

template void sytrs_solve_2x2<float>(wwr::wwrStream_t, float *, float *, int, int, const float *,
                                     const float *, const float *);
template void sytrs_solve_2x2<double>(wwr::wwrStream_t, double *, double *, int, int,
                                      const double *, const double *, const double *);
template void sytrs_solve_2x2<wwr::wwrFloatComplex>(wwr::wwrStream_t, wwr::wwrFloatComplex *,
                                                    wwr::wwrFloatComplex *, int, int,
                                                    const wwr::wwrFloatComplex *,
                                                    const wwr::wwrFloatComplex *,
                                                    const wwr::wwrFloatComplex *);
template void sytrs_solve_2x2<wwr::wwrDoubleComplex>(wwr::wwrStream_t, wwr::wwrDoubleComplex *,
                                                     wwr::wwrDoubleComplex *, int, int,
                                                     const wwr::wwrDoubleComplex *,
                                                     const wwr::wwrDoubleComplex *,
                                                     const wwr::wwrDoubleComplex *);

} // namespace calaman::device
