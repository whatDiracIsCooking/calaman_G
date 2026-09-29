// lacpy.cu
//
// The device-kernel half of calaman.lacpy: the single per-element copy stage.
// Shared unchanged between both backends -- under CUDA the .cu extension is all
// CMake needs, under HIP this directory's CMakeLists.txt forces LANGUAGE CXX
// back on so clang compiles it with -x hip.
//
// The copy is an index-per-thread map over the m*n elements of A, so it reuses
// wwr.extension.parallel_for rather than a hand-written launcher. A triangular
// region launches the full m*n grid and skips the elements outside it: correct,
// and simplest for a first cut. Mapping a triangular index range to (i, j)
// without the skipped threads is a possible later optimisation (see README).
#include "lacpy_bridge.h"

#include "extension/parallel_for/parallel_for.cuh"

#include <cstddef>

namespace calaman::device {

namespace {

// The integer region contract from lacpy_bridge.h, named for the predicate below.
constexpr int kRegionFull = 0;
constexpr int kRegionUpper = 1;
constexpr int kRegionLower = 2;

/// @brief Copies one column-major element A(i,j) -> B(i,j) if it is in-region
///
/// Members are const so the functor is not copy-assignable, which is what
/// parallel_for's device_functor concept checks for immutability. The linear
/// index splits column-major: j = idx / m, i = idx - j*m.
template<typename T>
struct lacpy_functor {
  const T *const a_;
  T *const b_;
  const std::size_t m_;
  const std::size_t lda_;
  const std::size_t ldb_;
  const int region_;

  __device__ void operator()(const std::size_t idx) const {
    const std::size_t j = idx / m_;
    const std::size_t i = idx - j * m_;

    // Upper is the diagonal and above (row <= col); lower is the diagonal and
    // below (row >= col). All threads take the same region_ branch, so the only
    // divergence is at the diagonal.
    bool in_region = true;
    if (region_ == kRegionUpper) {
      in_region = i <= j;
    } else if (region_ == kRegionLower) {
      in_region = i >= j;
    }

    if (in_region) {
      b_[i + j * ldb_] = a_[i + j * lda_];
    }
  }
};

} // namespace

template<typename T>
void launch_lacpy(const wwr::wwrStream_t stream, const int region, const std::size_t m,
                  const std::size_t n, const T *a, const std::size_t lda, T *b,
                  const std::size_t ldb) {
  const std::size_t count = m * n;
  if (count < 1) {
    return;
  }
  const lacpy_functor<T> functor{a, b, m, lda, ldb, region};
  wwr::extension::parallel_for(stream, count, functor);
}

// One per supported type, matching interface.cppm's extern template list and
// instantiations.cpp's -- all three lists cover the same types.
template void launch_lacpy<float>(wwr::wwrStream_t, int, std::size_t, std::size_t, const float *,
                                  std::size_t, float *, std::size_t);
template void launch_lacpy<double>(wwr::wwrStream_t, int, std::size_t, std::size_t, const double *,
                                   std::size_t, double *, std::size_t);

} // namespace calaman::device
