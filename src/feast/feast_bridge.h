/**
 * @file feast_bridge.h
 * @brief Shared types and device-launcher declarations for calaman.feast's
 *        interface units and its device-compiled translation unit
 *
 * Included by the module partitions in their GLOBAL MODULE FRAGMENT, and by
 * feast.cu directly -- the same split lacpy_bridge.h / gebal_bridge.h use: the
 * declarations live in the GMF, not the module purview, so a purview name's
 * module linkage cannot stop them binding to the definitions compiled in the
 * plain .cu translation unit. FeastContour / FeastStatus and kFeastMaxNodes are
 * ordinary external-linkage header entities, so every partition that names them
 * (and feast.cu) shares one definition.
 *
 * COMPLEX TYPES DO NOT APPEAR HERE, for the reason gebal_bridge.h spells out:
 * this header is parsed in host GMFs that cannot `import wwr.complex`, and
 * complex.h's complex builders are gated to a device pass. So FeastContour holds
 * the nodes/weights as plain real components (the kernels already do the complex
 * arithmetic from them by hand), and every launcher is generic in its complex
 * element type @c ComplexT and real component type @c RealT -- which the caller
 * spells as wwr::RealToComplexType<T> and T. feast.cu names the concrete
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

namespace calaman::device {

/// @brief Most quadrature nodes a contour can carry. FeastContour is a kernel
///        argument, so this bounds the size of that argument, not of any buffer.
inline constexpr int kFeastMaxNodes = 8;

/// @brief The quadrature of FEAST's contour integral: nodes Z_e and weights w_e,
///        stored as real components so this POD is nameable in a host GMF.
///
/// Defines the rational filter rho(lambda) = sum_e Re[ w_e / (Z_e - lambda) ];
/// see compute_quadrature.cppm. Only the first @c count entries are meaningful.
///
/// @tparam R Real floating-point type (float or double).
template<typename R>
struct FeastContour {
  R zr[kFeastMaxNodes]; ///< Re Z_e
  R zi[kFeastMaxNodes]; ///< Im Z_e
  R wr[kFeastMaxNodes]; ///< Re w_e
  R wi[kFeastMaxNodes]; ///< Im w_e
  int count;
};

/// @brief What one FEAST iteration leaves on the device for the host to decide on.
///
/// The solver's batched-LU and eigensolver info outputs point into this block,
/// and the solver's kernels fill in the rest, so one device-to-host copy of it
/// per iteration is the loop's only synchronization.
///
/// @tparam R Real floating-point type (float or double).
template<typename R>
struct FeastStatus {
  int lu_info[kFeastMaxNodes]; ///< getrfBatched, one per node; written once per solve
  int qr_info[2];              ///< geqrf, orgqr
  int eig_info;                ///< syevd
  int m;                       ///< Ritz values in [Emin, Emax]
  int lo;                      ///< where the first of them sat in syevd's ascending order
  R max_residual;              ///< largest relative residual among those m
};

/// @brief (Z_e I - A) for every node, into packed n x n blocks @p stride elements apart.
///
/// Reads only the @p lower (else upper) triangle of A, so the other may hold
/// anything -- the convention for a symmetric argument.
///
/// @tparam ComplexT Complex element type of the output; @tparam RealT A's real type.
template<typename ComplexT, typename RealT>
void feast_resolvents(wwr::wwrStream_t stream, bool lower, int n, const RealT *d_A, int lda,
                      FeastContour<RealT> contour, ComplexT *d_out, std::size_t stride);

/// @brief d_ptrs[e] = d_base + e * stride for e < count: the device pointer arrays
///        the batched BLAS routines take in place of a stride.
template<typename ComplexT>
void feast_pointer_array(wwr::wwrStream_t stream, ComplexT *d_base, std::size_t stride, int count,
                         ComplexT **d_ptrs);

/// @brief @p count complex copies of the real array @p d_Y, @p stride elements
///        apart, each with zero imaginary part: the right-hand sides of the
///        shifted solves, which the solve then overwrites.
template<typename ComplexT, typename RealT>
void feast_broadcast(wwr::wwrStream_t stream, std::size_t elems, const RealT *d_Y, int count,
                     ComplexT *d_out, std::size_t stride);

/// @brief d_out = sum_e Re(w_e X_e), the X_e being contour.count complex blocks
///        of @p elems elements, @p stride apart. The filter's quadrature sum.
template<typename ComplexT, typename RealT>
void feast_accumulate(wwr::wwrStream_t stream, std::size_t elems, FeastContour<RealT> contour,
                      const ComplexT *d_X, std::size_t stride, RealT *d_out);

/// @brief ||A||_1 of the symmetric matrix whose @p lower (else upper) triangle is
///        stored, into the device scalar @p d_norm. @p d_colsum: n scratch.
///
/// A NaN anywhere in the stored triangle makes the norm NaN rather than being
/// dropped by the maximum, so it surfaces in the residuals.
template<typename RealT>
void feast_sym_norm1(wwr::wwrStream_t stream, bool lower, int n, const RealT *d_A, int lda,
                     RealT *d_colsum, RealT *d_norm);

/// @brief Pick out the Ritz pairs inside [emin, emax] and rotate them to the front.
///
/// @p d_ritz must be ascending, as syevd returns it, so the pairs inside form one
/// contiguous run [lo, lo + m) -- found by binary search in every thread. Rotating
/// left by lo puts that run first in both @p d_lambda and the columns of
/// @p d_rotated, with the remaining pairs after it. Writes m and lo to the status.
///
/// @param d_vecs m0 x m0 eigenvectors from syevd, one per column.
template<typename RealT>
void feast_select(wwr::wwrStream_t stream, int m0, RealT emin, RealT emax, const RealT *d_ritz,
                  const RealT *d_vecs, RealT *d_lambda, RealT *d_rotated,
                  FeastStatus<RealT> *d_status);

/// @brief Relative residual of each of the leading status->m Ritz pairs,
///
///     ||A x - lambda x||_1 / ((||A||_1 + |lambda|) ||x||_1),
///
/// into @p d_residuals (zero past m), and their maximum into status->max_residual.
///
/// @param d_X  n x m0 Ritz vectors.
/// @param d_AX n x m0, A times them.
template<typename RealT>
void feast_residuals(wwr::wwrStream_t stream, int n, int m0, const RealT *d_X, const RealT *d_AX,
                     const RealT *d_lambda, const RealT *d_norm, RealT *d_residuals,
                     FeastStatus<RealT> *d_status);

} // namespace calaman::device
