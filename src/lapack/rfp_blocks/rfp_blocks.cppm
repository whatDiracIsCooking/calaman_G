/**
 * @file rfp_blocks.cppm
 * @brief calaman.rfp_blocks -- where the three BLAS-sized blocks of an n-by-n
 *        triangle sit inside its Rectangular Full Packed array
 *
 * The host-side counterpart of calaman::device::rfp_index (tri_index.cuh): the
 * same layout (src/lapack/trttf/README.md), described per BLOCK rather than per
 * element, for routines that run BLAS on RFP sub-blocks (?hfrk, ?sfrk, ?tfsm).
 * The triangle splits into a leading n1-by-n1 block A11, a trailing n2-by-n2
 * block A22 and the off-diagonal block between them; n1 and n2 are LAPACK's N1
 * and N2 (n1 = n/2 for Uplo::U, n - n/2 for Uplo::L). Each block is a
 * column-major sub-array of ARF given by an element offset and leading
 * dimension, and is stored either as itself or as its (conjugate) transpose --
 * the stored `uplo` is what a herk/trsm on that block is told.
 *
 * Pure constexpr host arithmetic: no device code, no allocation.
 *
 * Usage:
 *   import calaman.rfp_blocks;
 *   const calaman::RfpBlocks b = calaman::rfp_blocks(transr != Trans::N, uplo, n);
 *   // herk on A11: order b.a11.order, uplo b.a11.uplo, C = d_arf + b.a11.offset,
 *   // ldc = b.a11.ld; the off-diagonal block is b.off.rows x b.off.cols as stored
 */

export module calaman.rfp_blocks;

import std;
import calaman.common; // Uplo

namespace calaman {

export using calaman::Uplo;

/// @brief One diagonal block (A11 or A22) of the triangle, as stored in ARF
export struct RfpTriangle {
  std::size_t order;  ///< n1 for A11, n2 for A22
  Uplo uplo;          ///< The triangle of the stored sub-array that holds it
  bool transposed;    ///< Stored as the (conjugate) transpose of the logical block
  std::size_t offset; ///< Element offset of the sub-array's (0, 0) in ARF
  std::size_t ld;     ///< Leading dimension of the sub-array
};

/// @brief The off-diagonal block of the triangle, as stored in ARF
///
/// Logically A12 (n1 x n2) for Uplo::U and A21 (n2 x n1) for Uplo::L; when
/// @p transposed, ARF holds its (conjugate) transpose and rows/cols swap.
export struct RfpRectangle {
  std::size_t rows;   ///< Row count as stored
  std::size_t cols;   ///< Column count as stored
  bool transposed;    ///< Stored as the (conjugate) transpose of the logical block
  std::size_t offset; ///< Element offset of its (0, 0) in ARF
  std::size_t ld;     ///< Leading dimension
};

/// @brief The three blocks of an n-by-n triangle in RFP storage
export struct RfpBlocks {
  std::size_t n1;   ///< Order of A11 (rows/cols 0 .. n1-1 of the full matrix)
  std::size_t n2;   ///< Order of A22 (rows/cols n1 .. n-1)
  RfpTriangle a11;  ///< Leading diagonal block
  RfpTriangle a22;  ///< Trailing diagonal block
  RfpRectangle off; ///< A12 (Uplo::U) or A21 (Uplo::L)
};

/// @brief The block geometry of the @p uplo triangle of order @p n in RFP
///
/// @param transr true for the transposed layout (TRANSR = 'T' or 'C')
/// @param uplo   Which triangle ARF holds
/// @param n      Order of the full matrix; n == 0 gives all-empty blocks
export constexpr RfpBlocks rfp_blocks(const bool transr, const Uplo uplo, const std::size_t n) {
  const bool upper = uplo == Uplo::U;
  const std::size_t n1 = upper ? n / 2 : n - n / 2;
  const std::size_t n2 = n - n1;
  const std::size_t even = 1 - n % 2;
  // The TRANSR = 'N' array is (n + even) x ceil(n/2); 'T' is its transpose.
  const std::size_t rows_n = n + even;
  const std::size_t cols_n = n - n / 2;

  // (row, col) of each block's corner in the 'N' array, and whether the 'N'
  // layout stores it transposed: Upper keeps A12 and A22 in place and folds A11
  // below them; Lower keeps A11 and A21 in place and folds A22 above them.
  struct Place {
    std::size_t r;
    std::size_t c;
    bool t;
  };
  const Place p11 = upper ? Place{n1 + 1, 0, true} : Place{even, 0, false};
  const Place p22 = upper ? Place{n1, 0, false} : Place{0, n1 - n2, true};
  const Place poff = upper ? Place{0, 0, false} : Place{n1 + even, 0, false};

  const auto offset = [&](const Place p) {
    return transr ? p.c + p.r * cols_n : p.r + p.c * rows_n;
  };
  const std::size_t ld = transr ? cols_n : rows_n;
  const auto tri = [&](const std::size_t order, const Place p) {
    const bool t = p.t != transr;
    const Uplo stored = t ? (upper ? Uplo::L : Uplo::U) : uplo;
    return RfpTriangle{order, stored, t, offset(p), ld};
  };
  const std::size_t off_rows = upper ? n1 : n2;
  const std::size_t off_cols = upper ? n2 : n1;
  return RfpBlocks{n1, n2, tri(n1, p11), tri(n2, p22),
                   RfpRectangle{transr ? off_cols : off_rows, transr ? off_rows : off_cols, transr,
                                offset(poff), ld}};
}

// Spot checks against the C( ... ) offsets Reference-LAPACK's ZHFRK passes its
// herk/gemm calls (1-based there, 0-based here), one per parity x layout.
static_assert(rfp_blocks(false, Uplo::L, 5).a22.offset == 5 &&
              rfp_blocks(false, Uplo::L, 5).off.offset == 3 &&
              rfp_blocks(false, Uplo::L, 5).a22.uplo == Uplo::U);
static_assert(rfp_blocks(false, Uplo::U, 5).a11.offset == 3 &&
              rfp_blocks(false, Uplo::U, 5).a22.offset == 2 &&
              rfp_blocks(false, Uplo::U, 5).a11.uplo == Uplo::L);
static_assert(rfp_blocks(false, Uplo::L, 4).a11.offset == 1 &&
              rfp_blocks(false, Uplo::L, 4).off.offset == 3 &&
              rfp_blocks(false, Uplo::L, 4).a11.ld == 5);
static_assert(rfp_blocks(true, Uplo::L, 5).a22.offset == 1 &&
              rfp_blocks(true, Uplo::L, 5).off.offset == 9 &&
              rfp_blocks(true, Uplo::L, 5).off.rows == 3 &&
              rfp_blocks(true, Uplo::L, 5).a11.ld == 3);
static_assert(rfp_blocks(true, Uplo::U, 4).a11.offset == 6 &&
              rfp_blocks(true, Uplo::U, 4).a22.offset == 4 &&
              rfp_blocks(true, Uplo::U, 4).a11.uplo == Uplo::U &&
              rfp_blocks(true, Uplo::U, 4).off.ld == 2);

} // namespace calaman
