/**
 * @file diis_bridge.h
 * @brief Device-launcher declaration shared between calaman.diis's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT (a purview name gets
 * module linkage and could never bind to the plain-TU definition in diis.cu),
 * and by diis.cu directly. wwrStream_t arrives from runtime.h, the same type
 * wwr.runtime_api exports.
 */

#pragma once

#include <runtime.h>

namespace calaman::device {

/// @brief Solve the bordered DIIS system for the @p m coefficients in one block
///
/// Reads the upper triangle `gram[a*ld + c]` (a <= c < m) of the residual Gram,
/// scales B by its largest diagonal, solves [B -1; -1^T 0][c; l] = [0; -1] by
/// partial-pivot Gaussian elimination, and writes @p coeff (m elements). A
/// pivot below @p pivot_tol (relative to that scale), or an all-zero B, marks
/// the subspace singular: @p coeff becomes the unit selector at @p newest. Writes 1
/// (singular) or 0 to @p singular_out unless it is null. Enqueued on @p stream.
///
/// @tparam T Element type; instantiated for float, double
/// @pre 1 <= m <= ld <= kDiisMaxHistory, 0 <= newest < m
template<typename T>
void diis_solve(wwr::wwrStream_t stream, int m, int ld, int newest, const T *gram, T *coeff,
                int *singular_out, T pivot_tol);

} // namespace calaman::device
