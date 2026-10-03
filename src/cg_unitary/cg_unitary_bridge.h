/**
 * @file cg_unitary_bridge.h
 * @brief Device-launcher declarations and the sample cap, shared between
 *        calaman.cg_unitary's interface units and its device-compiled TU
 *
 * Included by buffer_size.cppm and geodesic_search.cppm in their GLOBAL MODULE
 * FRAGMENTS and by cg_unitary.cu directly -- the split nnls/laqps use: the
 * declarations live in the GMF, not the module purview, so a purview name's
 * module linkage cannot stop them binding to the definitions compiled in the
 * plain .cu translation unit.
 *
 * The geodesic line searches of Abrudan, Eriksson & Koivunen (Signal Processing
 * 89 (2009), Tables 1 and 2) take a handful of derivative samples as the
 * device-mode output of cuBLAS-equivalent dot products, then do a small amount
 * of scalar work on them: a Vandermonde solve (Table 1), a windowed DFT and a
 * complex root solve (Table 2), and a selection among the candidate step sizes.
 * All of it runs here, in single-block kernels, so a search synchronizes O(1)
 * times rather than once per sample. The arithmetic is trivial against the
 * O(n^3) products around it -- the point is not to move the data to the host.
 *
 * COMPLEX TYPES DO NOT APPEAR as concrete names here, for the reason
 * expm_bridge.h spells out: this header is parsed in a host GMF that cannot
 * `import wwr.complex`, and complex.h's device builders are gated to a device
 * pass. Every launcher is therefore GENERIC in the element type @c T, its real
 * component @c R, and (where a coefficient array is complex) a complex type
 * @c Cplx the caller supplies; cg_unitary.cu names the concrete
 * wwrFloatComplex / wwrDoubleComplex only in its explicit instantiations, in
 * device context. The launchers return void, like nnls's: a launch failure
 * surfaces at the next runtime copy/sync the caller already issues, and the
 * bridge cannot name wwrError_t (runtime.h gates it to the device pass too).
 *
 * wwrStream_t arrives from runtime.h, an include-only header, since a GMF cannot
 * import; it is the SAME type wwr.runtime_api exports. Reading the backend
 * define that header needs is why the module links wwr_backend PRIVATE -- see
 * this directory's CMakeLists.txt, the calaman.nnls arrangement.
 */

#pragma once

#include "runtime.h"

#include <cstddef>

namespace calaman {

/// @brief Largest number of derivative samples the single-block stages accept.
///
/// Matches the root finder's single-warp bound: N_DFT = 31 at the paper's
/// default K = 3, N_T = 10, and Table 1 needs only P + 1 <= 6.
inline constexpr int kCgMaxSamples = 32;

namespace device {

/// @brief Set v[0:n) to one in the element type (complex: (1, 0)).
///
/// The power-iteration warm start for skew_spectral_radius: the solver zeroes
/// the vector once per solve, and the first search reseeds it through here.
template<typename T>
void cg_seed_ones(wwr::wwrStream_t stream, T *d_v, int n);

/// @brief Coefficients of Table 1's approximating polynomial, from the samples.
///
/// The derivative samples arrive as raw dot products; the real part of each,
/// scaled by @p sample_scale, is the derivative (paper eq. 14). With mu_i = i h,
/// i = 0..P, the interpolation conditions sum_j a_j (i h)^j = Jhat'(mu_i) -
/// Jhat'(0) factor as M b = rhs with M[i][j] = i^j an integer matrix and
/// b_j = a_j h^j; solving the integer system then dividing out the powers of h
/// keeps the conditioning independent of h. a_0 is the sample at 0.
///
/// @tparam T Element type of the samples; @tparam R its real component type.
/// @param d_dots       Device, P + 1 dot-product results, sample order.
/// @param order        P, the polynomial order; 1 <= P < kCgMaxSamples.
/// @param h            Sample spacing T_mu / P.
/// @param sample_scale Multiplies the real part of each dot to give the
///                     derivative: 2 when minimizing, -2 when maximizing.
/// @param d_coeffs     Device out, P + 1 real coefficients, ascending degree.
template<typename T, typename R>
void cg_table1_coeffs(wwr::wwrStream_t stream, const T *d_dots, int order, R h, R sample_scale,
                      R *d_coeffs);

/// @brief Smallest positive real root of a real polynomial in (0, @p upper].
///
/// The first zero crossing of the fitted derivative, hence the first extremum
/// along the geodesic. Brackets a sign change on a fine grid of (0, upper] and
/// bisects it -- robust for the degree-<= 5 polynomials Table 1 produces, and
/// the grid's upper bound is exactly the interval beyond which a root would be
/// extrapolation anyway.
///
/// @param d_coeffs Device, @p order + 1 real coefficients, ascending degree.
/// @param order    Polynomial degree; 1 <= order <= 5.
/// @param upper    Upper bound of the bracket (the search interval T_mu).
/// @param d_mu     Device out, one real: the root, or 0 if none was bracketed.
/// @param d_found  Device out, one int: 1 if a root was bracketed, else 0.
template<typename R>
void cg_poly_smallest_positive_real_root(wwr::wwrStream_t stream, const R *d_coeffs, int order,
                                         R upper, R *d_mu, int *d_found);

/// @brief Hann-windowed centred DFT of Table 2's derivative samples (steps 8-10).
///
/// The window h(i) = 0.5 - 0.5 cos(2 pi (i + 1) / (N + 1)) is strictly positive
/// over i = 0..N-1, so it moves no zero of the derivative. The output is the
/// centred Fourier series c_k, k = -(N-1)/2 .. (N-1)/2, written at index
/// k + (N-1)/2 -- which, shifted by z^((N-1)/2), is directly the ascending-degree
/// coefficient array of an ordinary polynomial with the same roots.
///
/// @tparam T Element type of the samples; @tparam R its real component;
///         @tparam Cplx the complex type of the output coefficients.
/// @param d_dots      Device, @p num_samples dot-product results.
/// @param num_samples N_DFT; odd, and at most kCgMaxSamples.
/// @param sample_scale As for cg_table1_coeffs.
/// @param d_coeffs    Device out, @p num_samples complex coefficients.
template<typename T, typename R, typename Cplx>
void cg_hann_dft(wwr::wwrStream_t stream, const T *d_dots, int num_samples, R sample_scale,
                 Cplx *d_coeffs);

/// @brief Zero crossings of the reconstructed Fourier derivative, as arguments
///        in [0, 2 pi) (Table 2 step 11).
///
/// The @p num_coeffs centred coefficients define a real trig polynomial
/// D(theta) = sum_k c_k e^{i k theta} (real because the windowed samples are);
/// its zeros over (0, 2 pi) are the geodesic derivative's crossings in the angle
/// theta = 2 pi mu / T_DFT. Found by bracketing sign changes on a fine grid and
/// bisecting -- no complex root solve to diverge. Arguments ascending in
/// @p d_args, count in @p d_num_args, @p d_info 0 on success.
///
/// @tparam R Real component type; @tparam Cplx the coefficient complex type.
template<typename R, typename Cplx>
void cg_dft_root_args(wwr::wwrStream_t stream, const Cplx *d_coeffs, int num_coeffs, R *d_args,
                      int *d_num_args, int *d_info);

/// @brief Table 2's step-size selection, steps 12 and 13.
///
/// Root arguments map to step sizes by mu = arg T_DFT / (2 pi). With a descent
/// direction the derivative opens negative, so along the ordered crossings the
/// minima are the 1st, 3rd, 5th (zero-based even). Which to take is decided by
/// the sampled cost (step 13): the candidate nearest the best-improving sample.
///
/// @param d_args      Device, root arguments (ascending), count in @p d_num_args.
/// @param d_num_args  Device, how many arguments; may be zero.
/// @param d_cost      Device, @p num_samples sampled cost values.
/// @param num_samples N_DFT.
/// @param t_dft       The DFT interval length.
/// @param maximizing  1 to pick the largest sampled cost, 0 the smallest.
/// @param d_mu        Device out, one real: the chosen step size, or 0.
template<typename R>
void cg_select_dft_step(wwr::wwrStream_t stream, const R *d_args, const int *d_num_args,
                        const R *d_cost, int num_samples, R t_dft, int maximizing, R *d_mu);

/// @brief Real part of each dot product, scaled -- the sampled cost values.
///
/// Table 2 step 13 needs J at each sample; for the cost functions here the value
/// is itself a Frobenius inner product, so it arrives as a dot product like the
/// derivative does and only its real part is taken.
///
/// @tparam T Element type of the samples; @tparam R its real component type.
template<typename T, typename R>
void cg_real_parts(wwr::wwrStream_t stream, const T *d_dots, int count, R scale, R *d_out);

} // namespace device

} // namespace calaman
