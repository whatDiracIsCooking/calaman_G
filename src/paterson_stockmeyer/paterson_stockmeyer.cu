// paterson_stockmeyer.cu
//
// The device-kernel half of calaman.paterson_stockmeyer: building each block
//
//     B_j = c_(js) I + c_(js+1) A + ... + c_(js+m-1) A^(m-1)
//
// from the precomputed powers of A in a single fused pass. Everything else in
// the module is a host composition of wrapped BLAS gemms (the power bank and
// the outer Horner steps); this block build is the only genuinely per-element
// device work, so like horner.cu / laqps.cu it is launched through
// wwr.extension.parallel_for (a device-code header #included here, not a
// module). Shared unchanged between both backends, like lacpy.cu -- under CUDA
// the .cu extension is all CMake needs, under HIP this directory's
// CMakeLists.txt forces LANGUAGE CXX back on so clang compiles it with -x hip
// (which is also why there is no __CUDACC__ guard: it would empty the file, and
// with it the instantiations, on the HIP build).
//
// Forming a block with BLAS would be m separate axpy launches, every one a full
// read-modify-write pass over n^2 elements. This does it in one: each thread
// owns one element of B_j, walks the m terms, and writes once -- the same
// argument as horner's fused diagonal updates.
//
// The coefficients are read from a DEVICE pointer, which is what keeps the whole
// evaluation on the handle's stream without a host round-trip: a coefficient
// computed by an earlier kernel can be fed straight in. This is independent of
// the BLAS handle's pointer mode -- the kernel touches device memory directly;
// the gemm scalars are the host constants kOne/kZero.
//
// Neither the diagonal nor any power read touches the padding rows between n and
// ldb, so B may be a submatrix view of a larger allocation.
//
// Float and double only, matching the rest of calaman: complex is the deliberate
// later extension calaman.common's constants.h documents.
#include "paterson_stockmeyer_bridge.h"

#include <extension/parallel_for/parallel_for.cuh>

#include <cstddef>

namespace calaman::device {

namespace {

/// @brief Per-element functor for one Paterson-Stockmeyer block, via parallel_for
///
/// Trivially copyable with only const members, as device_functor requires (a
/// mutable member would be UB in the grid-constant copy). One thread per element
/// of the n-by-n block, indexed column-major so consecutive threads walk
/// consecutive addresses in every power read -- a coalesced access down each
/// column.
template<typename T>
struct BuildBlockFunctor {
  T *const d_B_;
  const std::size_t ldb_;
  const std::size_t n_;
  const T *const d_A_;
  const std::size_t lda_;
  const T *const d_powers_;
  const std::size_t ld_powers_;
  const std::size_t stride_;
  const T *const d_coeffs_;
  const int num_terms_;

  __device__ void operator()(const std::size_t idx) const {
    const std::size_t i = idx % n_;
    const std::size_t j = idx / n_;

    // k = 0: A^0 is the identity, so it contributes only on the diagonal and is
    // never read from memory.
    T acc = (i == j) ? d_coeffs_[0] : T{0};

    // k = 1: A itself, which keeps the caller's leading dimension.
    if (num_terms_ > 1) {
      acc = d_coeffs_[1] * d_A_[i + j * lda_] + acc;
    }

    // k >= 2: the uniform power bank, A^2 first.
    for (int k = 2; k < num_terms_; ++k) {
      const T *power = d_powers_ + static_cast<std::size_t>(k - 2) * stride_;
      acc = d_coeffs_[k] * power[i + j * ld_powers_] + acc;
    }

    d_B_[i + j * ldb_] = acc;
  }
};

} // namespace

template<typename T>
void paterson_stockmeyer_build_block(const wwr::wwrStream_t stream, T *const d_B, const int ldb,
                                     const int n, const T *const d_A, const int lda,
                                     const T *const d_powers, const int ld_powers,
                                     const std::size_t stride, const T *const d_coeffs,
                                     const int num_terms) {
  if (n < 1 || num_terms < 1) {
    return;
  }
  const std::size_t order = static_cast<std::size_t>(n);
  const BuildBlockFunctor<T> functor{d_B,
                                     static_cast<std::size_t>(ldb),
                                     order,
                                     d_A,
                                     static_cast<std::size_t>(lda),
                                     d_powers,
                                     static_cast<std::size_t>(ld_powers),
                                     stride,
                                     d_coeffs,
                                     num_terms};
  wwr::extension::parallel_for<std::size_t>(stream, order * order, functor);
}

// One per supported type, matching paterson_stockmeyer_bridge.h's declaration
// and the module's use sites -- float and double, as the evaluation is
// templated over.
template void paterson_stockmeyer_build_block<float>(wwr::wwrStream_t, float *, int, int,
                                                     const float *, int, const float *, int,
                                                     std::size_t, const float *, int);
template void paterson_stockmeyer_build_block<double>(wwr::wwrStream_t, double *, int, int,
                                                      const double *, int, const double *, int,
                                                      std::size_t, const double *, int);

} // namespace calaman::device
