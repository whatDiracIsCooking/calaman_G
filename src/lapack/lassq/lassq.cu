// lassq.cu
//
// The device half of calaman.lassq: the ?lassq scaled sum of squares of a
// strided vector, folded into the caller's (scale, sumsq) pair in ONE
// single-block launch.
//
// THE THREE ACCUMULATORS are ?lassq's own (LAPACK >= 3.10): abig takes squares
// of entries above kTbig scaled DOWN by kSbig, asml those below kTsml scaled UP
// by kSsml, amed the mid-range band unscaled -- so no square over- or
// underflows. Each thread folds a contiguous chunk into its own three,
// block_reduce sums each across the block, and thread 0 runs ?lassq's combine
// step and writes the pair back.
//
// NOTBIG IS abig == 0. The reference carries a sequential `notbig` flag that
// stops feeding asml once an entry above kTbig has been seen; a parallel fold
// has no "once", and needs none: asml is read only in the combine's
// `else if (asml > 0)` arm, which abig > 0 already excludes, and abig > 0 holds
// exactly when some entry exceeded kTbig.
//
// A complex entry contributes |Re| and |Im| as two separate entries, as ZLASSQ's
// loop does.
//
// One block is deliberate, matching lanst.cu: the fold is memory-bound over a
// single vector, and the (scale, sumsq) pair is read-modify-written, which one
// block does without a second pass or a grid-wide barrier.
#include "lassq_bridge.h"

#include "common/block_reduce.cuh"
#include "common/elem_ops.cuh"

#include <cstddef>
#include <type_traits>

namespace calaman::device {

namespace {

constexpr unsigned int kBlock = 256;

/// @brief ?lassq's band thresholds and scale factors (LAPACK la_constants.f90)
///
/// Exact powers of two under IEEE-754, spelled as hex float literals and
/// written out rather than computed: a device pass has no <limits>. The
/// comments give the radix-model exponent each one comes from.
template<typename R>
struct LassqConstants;

template<>
struct LassqConstants<float> {
  static constexpr float kTsml = 0x1p-63F;  // 2^ceil((minexponent-1)/2)
  static constexpr float kTbig = 0x1p+52F;  // 2^floor((maxexponent-digits+1)/2)
  static constexpr float kSsml = 0x1p+75F;  // 2^-floor((minexponent-digits)/2)
  static constexpr float kSbig = 0x1p-76F;  // 2^-ceil((maxexponent+digits-1)/2)
};

template<>
struct LassqConstants<double> {
  static constexpr double kTsml = 0x1p-511;
  static constexpr double kTbig = 0x1p+486;
  static constexpr double kSsml = 0x1p+537;
  static constexpr double kSbig = 0x1p-538;
};

/// @brief Add |entry|^2 to whichever of the three accumulators keeps it in range
///
/// @p ax is already an absolute value. A NaN compares false against both
/// thresholds and so lands in @p amed, which propagates it -- DLASSQ's own path.
template<typename R>
__device__ void fold_entry(const R ax, R &asml, R &amed, R &abig) {
  using C = LassqConstants<R>;
  if (ax > C::kTbig) {
    const R t = ax * C::kSbig;
    abig += t * t;
  } else if (ax < C::kTsml) {
    const R t = ax * C::kSsml;
    asml += t * t;
  } else {
    amed += ax * ax;
  }
}

/// @brief [kernel] Fold the ?lassq sum of squares of x into (*d_scale, *d_sumsq)
template<typename T, typename R>
__global__ void lassq_kernel(const std::size_t n, const T *const x, const std::ptrdiff_t ix0,
                             const std::ptrdiff_t incx, R *const d_scale, R *const d_sumsq) {
  using C = LassqConstants<R>;

  // Every thread reads the same two scalars, so the branches below are uniform
  // -- which is what lets them precede the block-wide barriers in block_reduce.
  // Only thread 0 writes, after the last of those barriers.
  R scl = *d_scale;
  R sumsq = *d_sumsq;

  // DLASSQ's quick return and normalization, in its order: a NaN in either
  // scalar leaves both untouched, and a pair representing zero is canonicalized.
  if (scl != scl || sumsq != sumsq) {
    return;
  }
  if (sumsq == R{0}) {
    scl = R{1};
  }
  if (scl == R{0}) {
    scl = R{1};
    sumsq = R{0};
  }
  if (n == 0) {
    if (threadIdx.x == 0) {
      *d_scale = scl;
      *d_sumsq = sumsq;
    }
    return;
  }

  const std::size_t chunk = (n + kBlock - 1) / kBlock;
  const unsigned int nactive = static_cast<unsigned int>((n + chunk - 1) / chunk);
  const std::size_t lo = threadIdx.x * chunk;
  const std::size_t hi = lo + chunk < n ? lo + chunk : n;

  R asml = R{0};
  R amed = R{0};
  R abig = R{0};
  for (std::size_t i = lo; i < hi; ++i) {
    const T v = x[ix0 + static_cast<std::ptrdiff_t>(i) * incx];
    if constexpr (std::is_same_v<T, R>) {
      fold_entry(elem_ops<R>::modulus(v), asml, amed, abig);
    } else {
      // ZLASSQ's loop: the two components are two entries of the real vector
      // whose sum of squares |z|^2 adds up to.
      fold_entry(wwr::fabs(elem_ops<T>::real_part(v)), asml, amed, abig);
      fold_entry(wwr::fabs(elem_ops<T>::imag_part(v)), asml, amed, abig);
    }
  }

  // Three folds of the same R reuse block_reduce's one shared buffer, serialized
  // by its own leading and trailing barriers. Every thread must reach all three.
  asml = block_reduce<kBlock>(asml, AddOp{}, nactive);
  amed = block_reduce<kBlock>(amed, AddOp{}, nactive);
  abig = block_reduce<kBlock>(abig, AddOp{}, nactive);

  if (threadIdx.x != 0) {
    return;
  }

  // Put the incoming sum of squares into one of the accumulators (DLASSQ's
  // second stage). `abig == R{0}` is the reference's notbig -- see the header.
  if (sumsq > R{0}) {
    const R ax = scl * wwr::sqrt(sumsq);
    if (ax > C::kTbig) {
      const R t = scl * C::kSbig;
      abig += t * t * sumsq;
    } else if (ax < C::kTsml) {
      if (abig == R{0}) {
        const R t = scl * C::kSsml;
        asml += t * t * sumsq;
      }
    } else {
      amed += scl * scl * sumsq;
    }
  }

  // Combine whichever accumulators were used, verbatim from DLASSQ. `amed > 0`
  // already covers its `amed > HUGE` arm (an infinity is > 0); the `!=` catches
  // a NaN, which no comparison would.
  if (abig > R{0}) {
    if (amed > R{0} || amed != amed) {
      abig += (amed * C::kSbig) * C::kSbig;
    }
    scl = R{1} / C::kSbig;
    sumsq = abig;
  } else if (asml > R{0}) {
    if (amed > R{0} || amed != amed) {
      const R rooted_med = wwr::sqrt(amed);
      const R rooted_sml = wwr::sqrt(asml) / C::kSsml;
      const R ymax = rooted_sml > rooted_med ? rooted_sml : rooted_med;
      const R ymin = rooted_sml > rooted_med ? rooted_med : rooted_sml;
      scl = R{1};
      sumsq = ymax * ymax * (R{1} + (ymin / ymax) * (ymin / ymax));
    } else {
      scl = R{1} / C::kSsml;
      sumsq = asml;
    }
  } else {
    scl = R{1};
    sumsq = amed;
  }

  *d_scale = scl;
  *d_sumsq = sumsq;
}

} // namespace

template<typename T, typename R>
void lassq(const wwr::wwrStream_t stream, const std::size_t n, const T *const x, const int incx,
           R *const d_scale, R *const d_sumsq) {
  // ?lassq's ix: a negative incx starts at the far end so the walk runs down
  // through the same elements a positive incx runs up through.
  const std::ptrdiff_t step = incx;
  const std::ptrdiff_t ix0 =
      incx < 0 ? -static_cast<std::ptrdiff_t>(n == 0 ? 0 : n - 1) * step : std::ptrdiff_t{0};

  lassq_kernel<T, R><<<1, kBlock, 0, stream>>>(n, x, ix0, step, d_scale, d_sumsq);
}

// One per supported type, matching lassq_bridge.h and interface.cppm's
// extern-template list -- a type added here but not there links against nothing.
template void lassq<float, float>(wwr::wwrStream_t, std::size_t, const float *, int, float *,
                                  float *);
template void lassq<double, double>(wwr::wwrStream_t, std::size_t, const double *, int, double *,
                                    double *);
template void lassq<wwr::wwrFloatComplex, float>(wwr::wwrStream_t, std::size_t,
                                                 const wwr::wwrFloatComplex *, int, float *,
                                                 float *);
template void lassq<wwr::wwrDoubleComplex, double>(wwr::wwrStream_t, std::size_t,
                                                   const wwr::wwrDoubleComplex *, int, double *,
                                                   double *);

} // namespace calaman::device
