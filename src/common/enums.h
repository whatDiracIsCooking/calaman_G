/**
 * @file enums.h
 * @brief Scoped enums for the LAPACK-shaped selector arguments (Jobz, Uplo,
 *        Trans, Side, Diag, Range, JobSvd, Region) and the diff_norm Norm
 *        selector, as a header shareable by device .cu code and module GMFs alike
 *
 * A plain header, not a module unit, for the same reason constants.h and
 * align_up.h are: a scoped enum is a compile-time tag that both a device
 * translation unit (a .cu, a plain TU that cannot `import`) and a module's
 * global module fragment want to name. The sibling enums.cppm includes this in
 * its GMF and re-exports the names, so importers of calaman.common see the same
 * `calaman::` spellings as #includers.
 *
 * Backend-neutral by construction (CLAUDE.md): most of these are this project's
 * own strongly-typed stand-ins for the single-char LAPACK selectors, so a call
 * site passes `Uplo::L` rather than a bare `'L'`. The mapping from an enumerator
 * to the char (or vendor constant) a backend wants is a call-site concern and
 * does not belong here. Norm is the exception -- not a LAPACK char but the
 * reduction diff_norm selects (ell_1 / ell_2 / ell_inf); it lives here because it
 * is shared vocabulary rather than a diff_norm-private detail.
 *
 * Region is the near-twin of Uplo kept deliberately apart: `?lacpy`/`?laset`
 * overload their `uplo` char with a third "all of the matrix" case that no
 * symmetric routine has, so Region carries that third enumerator while Uplo
 * stays exactly two-valued for Cholesky, the symmetric eigensolvers and the
 * symmetric-norm routines.
 *
 * Consumers #include this by its root-relative path, "common/enums.h".
 */

#pragma once

#include <cstdint>

namespace calaman {

// ========================================================================
// LAPACK selector enums
// ========================================================================

/// @brief Whether an eigen/SVD routine computes vectors (LAPACK `jobz`).
enum class Jobz : std::uint8_t {
    N, ///< Eigen/singular values only; no vectors.
    V, ///< Compute eigen/singular values and vectors.
};

/// @brief Which triangle of a symmetric/Hermitian matrix is referenced
///        (LAPACK `uplo`).
enum class Uplo : std::uint8_t {
    U, ///< Upper triangle.
    L, ///< Lower triangle.
};

/// @brief Which region of a general matrix a copy/fill touches -- LAPACK's
///        `?lacpy`/`?laset` UPLO, whose else-case copies/sets the whole matrix.
///
/// Distinct from Uplo on purpose: those two routines overload the `uplo` char
/// with a third "all of the matrix" meaning, which no symmetric routine shares.
enum class Region : std::uint8_t {
    U, ///< Upper triangle/trapezoid: the diagonal and above.
    L, ///< Lower triangle/trapezoid: the diagonal and below.
    A, ///< All of the matrix: the whole m-by-n rectangle.
};

/// @brief Whether and how an operand is transposed (LAPACK `trans`).
enum class Trans : std::uint8_t {
    N, ///< No transpose: op(A) = A.
    T, ///< Transpose: op(A) = A^T.
    C, ///< Conjugate transpose: op(A) = A^H.
};

/// @brief Which side a (triangular/orthogonal) operand multiplies from
///        (LAPACK `side`).
enum class Side : std::uint8_t {
    L, ///< From the left: op(A) * B.
    R, ///< From the right: B * op(A).
};

/// @brief Whether a triangular matrix has a unit diagonal (LAPACK `diag`).
enum class Diag : std::uint8_t {
    N, ///< Non-unit: the diagonal is referenced.
    U, ///< Unit: the diagonal is assumed to be all ones.
};

/// @brief Which eigenvalues a selective eigensolver computes (LAPACK `range`).
enum class Range : std::uint8_t {
    A, ///< All eigenvalues.
    V, ///< Those in a half-open value interval (vl, vu].
    I, ///< Those with index in [il, iu].
};

/// @brief How much of U/V^T an SVD computes (LAPACK `jobu`/`jobvt`).
enum class JobSvd : std::uint8_t {
    A, ///< All columns/rows of U/V^T.
    S, ///< The first min(m, n) columns/rows (the reduced factor).
    O, ///< Overwrite the input with the singular vectors.
    N, ///< No columns/rows computed.
};

/// @brief Which norm of the difference diff_norm reports
enum class Norm : std::uint8_t {
    l1,  ///< ell_1: the sum of magnitudes, sum |y_i - x_i| (BLAS asum)
    l2,  ///< ell_2: the Euclidean norm, sqrt(sum |y_i - x_i|^2) (BLAS nrm2)
    inf, ///< ell_inf: the largest magnitude, max |y_i - x_i| (BLAS iamax + a read)
};

} // namespace calaman
