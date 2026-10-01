// horner.cu
//
// The device-kernel half of calaman.horner: the two diagonal updates of the
// Horner recurrence. Everything else in horner is a host composition of one
// wrapped BLAS gemm per step; these are the only genuinely per-element device
// work, so like laqp2.cu / laqps.cu they are launched through
// wwr.extension.parallel_for (a device-code header #included here, not a
// module). Shared unchanged between both backends, like lacpy.cu -- under CUDA
// the .cu extension is all CMake needs, under HIP this directory's
// CMakeLists.txt forces LANGUAGE CXX back on so clang compiles it with -x hip.
//
// Both updates read the coefficient from a DEVICE pointer, which is what keeps
// the whole evaluation on the handle's stream without a host round-trip: a
// coefficient computed by an earlier kernel can be fed straight in. This is
// independent of the BLAS handle's pointer mode -- these kernels touch device
// memory directly; the gemm scalars are the host constants kOne/kZero.
//
// Neither kernel touches the padding rows between n and ldp, so P may be a
// submatrix view of a larger allocation.
//
// Float and double only, matching the rest of calaman: complex is the deliberate
// later extension calaman.common's constants.h documents.
#include "horner_bridge.h"

#include "extension/parallel_for/parallel_for.cuh"

#include <cstddef>

namespace calaman::device {

namespace {

/// @brief Per-element functor for P := alpha * I, invoked by parallel_for
///
/// Trivially copyable with only const members, as device_functor requires (a
/// mutable member would be UB in the grid-constant copy). One thread per element
/// of the n-by-n block, indexed column-major so consecutive threads write
/// consecutive addresses -- a coalesced store down each column.
template<typename T>
struct HornerSetScaledIdentityFunctor {
  T *const d_P_;
  const std::size_t n_;
  const std::size_t ldp_;
  const T *const d_alpha_;

  __device__ void operator()(const std::size_t k) const {
    const std::size_t i = k % n_;
    const std::size_t j = k / n_;
    d_P_[j * ldp_ + i] = (i == j) ? *d_alpha_ : T{0};
  }
};

/// @brief Per-diagonal functor for P(i, i) += alpha, invoked by parallel_for
///
/// One thread per diagonal entry; off-diagonal entries are left untouched, so
/// this composes with the gemm that produced them.
template<typename T>
struct HornerAddScaledIdentityFunctor {
  T *const d_P_;
  const std::size_t ldp_;
  const T *const d_alpha_;

  __device__ void operator()(const std::size_t i) const {
    // (i, i) sits at i + i * ldp = i * (ldp + 1).
    d_P_[i * (ldp_ + 1)] += *d_alpha_;
  }
};

} // namespace

template<typename T>
void horner_set_scaled_identity(const wwr::wwrStream_t stream, T *const d_P, const int n,
                                const int ldp, const T *const d_alpha) {
  if (n < 1) {
    return;
  }
  const std::size_t order = static_cast<std::size_t>(n);
  const HornerSetScaledIdentityFunctor<T> functor{d_P, order, static_cast<std::size_t>(ldp),
                                                   d_alpha};
  wwr::extension::parallel_for<std::size_t>(stream, order * order, functor);
}

template<typename T>
void horner_add_scaled_identity(const wwr::wwrStream_t stream, T *const d_P, const int n,
                                const int ldp, const T *const d_alpha) {
  if (n < 1) {
    return;
  }
  const HornerAddScaledIdentityFunctor<T> functor{d_P, static_cast<std::size_t>(ldp), d_alpha};
  wwr::extension::parallel_for<std::size_t>(stream, static_cast<std::size_t>(n), functor);
}

// One per supported type, matching horner_bridge.h's declarations and the
// module's use sites -- float and double, as the evaluation is templated over.
template void horner_set_scaled_identity<float>(wwr::wwrStream_t, float *, int, int,
                                                const float *);
template void horner_set_scaled_identity<double>(wwr::wwrStream_t, double *, int, int,
                                                 const double *);

template void horner_add_scaled_identity<float>(wwr::wwrStream_t, float *, int, int,
                                                const float *);
template void horner_add_scaled_identity<double>(wwr::wwrStream_t, double *, int, int,
                                                 const double *);

} // namespace calaman::device
