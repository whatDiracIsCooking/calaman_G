/**
 * @file expm_bridge.h
 * @brief Pade coefficient tables and device-launcher declarations shared
 *        between calaman.expm's interface unit and its device-compiled
 *        translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT, and by expm.cu
 * directly -- the same split lacpy_bridge.h / gebal_bridge.h / horner_bridge.h
 * use: the declarations live in the GMF, not the module purview, so a purview
 * name's module linkage cannot stop them binding to the definitions compiled in
 * the plain .cu translation unit.
 *
 * Two kinds of thing live here. The Pade COEFFICIENT TABLES and the little
 * constexpr helpers over them (pade_coeffs, pade_num_powers, pade_num_gemms,
 * kPadeDegrees) are pure host arithmetic the interface reads to
 * drive the ladder; they name no device and no complex type, so a host GMF parse
 * is happy with them. The KERNEL LAUNCHERS are the four genuinely per-element
 * pieces of expm -- the fused Pade evaluation, the numerator/denominator split,
 * and the two halves of the 1-norm -- reached
 * through wwr.extension.parallel_for / calaman.reduce_columns from expm.cu.
 *
 * COMPLEX TYPES DO NOT APPEAR HERE, for the reason complex_cast_bridge.h spells
 * out: this header is parsed in a host GMF that cannot `import wwr.complex`, and
 * complex.h's complex builders are gated to a device pass. So every launcher is
 * generic in the element type @c T and its real component type @c R, which the
 * interface spells as wwr::ComplexToRealType<T>; the .cu names the concrete
 * wwrFloatComplex / wwrDoubleComplex only in its explicit instantiations, in
 * device context.
 *
 * wwrStream_t arrives from runtime.h, an include-only header rather than an
 * `import`, since a GMF cannot import; it is the SAME type wwr.runtime_api
 * exports, so the module passes its handle's stream straight through. Reading
 * the backend define that header needs is why the module links wwr_backend
 * PRIVATE -- see this directory's CMakeLists.txt.
 */

#pragma once

#include "runtime.h"

#include <cstddef>

namespace calaman {

// ========================================================================
// Diagonal Pade coefficients for exp(x)
// ========================================================================
//
//   p_m(x) = sum_{j=0}^{m} b_j x^j        q_m(x) = p_m(-x)
//   b_j    = (2m-j)! * m! / ( (2m)! * j! * (m-j)! )
//
// so that r_m(x) = q_m(x)^{-1} p_m(x) matches exp(x) through order x^(2m).
// b_0 = 1 for every m, so the tables below are already normalized; a common
// factor would cancel in r_m anyway. The exact rationals are spelled out rather
// than reduced to decimal so the tables can be checked by eye against the
// formula above.

inline constexpr double kPade3[4] = {
    1.0, 1.0 / 2.0, 1.0 / 10.0, 1.0 / 120.0,
};

inline constexpr double kPade5[6] = {
    1.0, 1.0 / 2.0, 1.0 / 9.0, 1.0 / 72.0, 1.0 / 1008.0, 1.0 / 30240.0,
};

inline constexpr double kPade7[8] = {
    1.0,          1.0 / 2.0,     3.0 / 26.0,     5.0 / 312.0,
    5.0 / 3432.0, 1.0 / 11440.0, 1.0 / 308880.0, 1.0 / 17297280.0,
};

inline constexpr double kPade9[10] = {
    1.0,          1.0 / 2.0,      2.0 / 17.0,      7.0 / 408.0,       7.0 / 4080.0,
    1.0 / 8160.0, 1.0 / 159120.0, 1.0 / 4455360.0, 1.0 / 196035840.0, 1.0 / 17643225600.0,
};

inline constexpr double kPade13[14] = {
    1.0,
    1.0 / 2.0,
    3.0 / 25.0,
    11.0 / 600.0,
    11.0 / 5520.0,
    3.0 / 18400.0,
    1.0 / 96600.0,
    1.0 / 1932000.0,
    1.0 / 48944000.0,
    1.0 / 1585785600.0,
    1.0 / 67395888000.0,
    1.0 / 3953892096000.0,
    1.0 / 355850288640000.0,
    1.0 / 64764752532480000.0,
};

/// @brief The Pade degrees the expm ladder may choose from, ascending.
inline constexpr int kPadeDegrees[5] = {3, 5, 7, 9, 13};
inline constexpr int kNumPadeDegrees = 5;

/// @brief The coefficient table for degree @p m, or nullptr if m is not on the ladder.
constexpr const double *pade_coeffs(const int m) {
  switch (m) {
  case 3:
    return kPade3;
  case 5:
    return kPade5;
  case 7:
    return kPade7;
  case 9:
    return kPade9;
  case 13:
    return kPade13;
  default:
    return nullptr;
  }
}

/// @brief Powers of A held at once when evaluating r_m: A^2, A^4, ... A^(2*np).
///
/// For m in {3,5,7,9} the even/odd split needs every even power up to m-1, so
/// np = (m-1)/2. Degree 13 is the exception: its nested form reaches only A^6
/// and recovers the higher terms from two extra products (see pade() in
/// interface.cppm), which is what keeps it at six matrix products instead of
/// seven.
constexpr int pade_num_powers(const int m) { return (m == 13) ? 3 : (m - 1) / 2; }

/// @brief Matrix products r_m costs: the power bank, plus the nested products.
///
/// np - 1 products build A^4 .. A^(2*np) once A^2 is formed, 1 forms A^2 itself,
/// and 1 more forms U = A * W. Degree 13 adds the two nested products.
constexpr int pade_num_gemms(const int m) {
  return pade_num_powers(m) + 1 + ((m == 13) ? 2 : 0);
}

namespace device {

/// @brief Fused evaluation of two real-coefficient polynomials in the same powers
///
/// In one pass over the supplied powers this writes
///
///   V = cv[0]*I + cv[1]*P1 + cv[2]*P2 + ... + cv[np]*P_np
///   W = cw[0]*I + cw[1]*P1 + cw[2]*P2 + ... + cw[np]*P_np
///
/// Fusing matters: expressed with BLAS this is 2*np separate axpy/geam launches,
/// each re-reading a full n*n matrix. Here every power is read exactly once and
/// both results are produced together. Every Pade degree on the ladder is built
/// from this one launcher: for m in {3,5,7,9} it produces the even and odd
/// halves of the numerator in a single call; for m = 13 it is called twice, each
/// call pairing one nested inner combination with the matching tail (a zero
/// constant term simply contributes nothing on the diagonal).
///
/// @tparam T Element type; @tparam R its real component type (cv/cw are real)
/// @param num_powers How many of d_P1..d_P4 are read; 1 to 4. Pointers past that
///                   count are never dereferenced and may be null.
/// @param cv         Host array of num_powers + 1 real coefficients for V
/// @param cw         Host array of num_powers + 1 real coefficients for W
///
/// All matrices are n x n column-major with leading dimension n.
template<typename T, typename R>
void pade_even_odd(wwr::wwrStream_t stream, int n, int num_powers, const T *d_P1, const T *d_P2,
                   const T *d_P3, const T *d_P4, const R *cv, const R *cw, T *d_V, T *d_W);

/// @brief Fused split of the Pade numerator and denominator
///
/// Given U = A*W and V, writes in a single pass
///   P = V + U   (the numerator   p(A), into d_P with leading dimension @p ldp)
///   Q = V - U   (the denominator q(A) = p(-A), into d_Q, leading dimension n)
///
/// @note d_Q may alias d_V. Each element is read by the one thread that writes
///       it, so the overwrite is safe, and it is what lets the denominator be
///       factored in place of the even half.
template<typename T>
void pade_split(wwr::wwrStream_t stream, int n, const T *d_U, const T *d_V, T *d_P, int ldp,
                T *d_Q);

/// @brief Column absolute sums: d_colsum[j] = sum_i |A(i, j)| for each column j
///
/// Complex elements use the TRUE modulus, not the |Re| + |Im| surrogate: this is
/// the induced 1-norm. Defers to calaman.reduce_columns (a header-only, no-Thrust
/// segmented reduce) with |.| as the pre-transform and + as the fold.
/// Asynchronous; results are valid once @p stream completes.
///
/// @tparam T Element type; @tparam R its real component type (the sum is real)
template<typename T, typename R>
void abs_colsums(wwr::wwrStream_t stream, int n, const T *d_A, int lda, R *d_colsum);

/// @brief Reduce n non-negative reals to their maximum, in place at d_values[0]
///
/// A NaN anywhere in the input wins and is left in d_values[0], so a caller can
/// reject a matrix holding one rather than proceeding on a plausible-looking
/// norm. Asynchronous: the single-block reduction is enqueued on @p stream and
/// the caller reads d_values[0] back once the stream has drained.
///
/// @tparam R Real type; instantiated for float, double
/// @param d_values Device array of n reals; OVERWRITTEN -- the reduction runs in
///                 place and leaves its result in d_values[0]
template<typename R>
void max_reduce(wwr::wwrStream_t stream, int n, R *d_values);

} // namespace device

} // namespace calaman
