/**
 * @file lantr_bridge.h
 * @brief Device-launcher declaration shared between the calaman.lantr,
 *        calaman.lantp and calaman.lantb interface units and their one
 *        device-compiled TU
 *
 * Included in each interface's GLOBAL MODULE FRAGMENT (a purview name gets
 * module linkage and could never bind to the plain-TU definition in lantr.cu)
 * and by lantr.cu directly -- lansb_bridge.h's split; calaman.lantp and
 * calaman.lantb reach it root-relative as "lapack/lantr/lantr_bridge.h". R is
 * T's real component type, a second parameter so this GMF header needs no
 * fp_types.h.
 */

#pragma once

#include "common/enums.h"
#include <runtime.h>

#include <cstddef>
#include <cstdint>

namespace calaman::device {

/// @brief How the stored triangle is laid out: the one difference between the
///        three triangular norms
enum class TriStorage : std::uint8_t {
  full,   ///< ?lantr: A(i,j) at A[i + j*ld], m-by-n trapezoid
  packed, ///< ?lantp: A(i,j) at AP[packed_index(n, i, j)]; ld unused
  band,   ///< ?lantb: A(i,j) at AB[(k+i-j or i-j) + j*ld], k off-diagonals
};

/// @brief Enqueue the ?lantr / ?lantp / ?lantb norm of a triangular matrix
///
/// Two launches (per-line partials into @p d_scratch, then one fold into
/// @p d_result); returns without synchronizing. Assumes @p m, @p n >= 1.
///
/// @tparam T Element type; float, double, wwrFloatComplex, wwrDoubleComplex
/// @tparam R Real component type of T -- the type of the scratch and the result
/// @param stream    Stream the launches are enqueued on; all pointers live on its device
/// @param which     Which norm to compute
/// @param uplo      Upper or lower trapezoid
/// @param diag      Diag::U counts each diagonal entry as 1 and does not read it
/// @param storage   Layout of @p d_A
/// @param m         Rows of A (== n for packed and band)
/// @param n         Columns of A
/// @param k         Off-diagonals of a band A; ignored otherwise
/// @param d_A       The stored triangle
/// @param ld        Leading dimension of @p d_A (full, band); ignored for packed
/// @param d_result  Device scalar receiving the norm
/// @param d_scratch Device scratch of m (MatrixNorm::inf) or n (otherwise) elements
template<typename T, typename R>
void lantr(wwr::wwrStream_t stream, MatrixNorm which, Uplo uplo, Diag diag, TriStorage storage,
           std::size_t m, std::size_t n, std::size_t k, const T *d_A, std::size_t ld, R *d_result,
           R *d_scratch);

} // namespace calaman::device
