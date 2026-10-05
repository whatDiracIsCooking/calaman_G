/**
 * @file ladiv_bridge.h
 * @brief Device-launcher declaration shared between calaman.ladiv's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT and by ladiv.cu
 * directly -- the split lartg_bridge.h uses: the declaration lives in the GMF,
 * not the module purview, so a purview name's module linkage cannot stop it
 * binding to the definition compiled in the plain .cu translation unit.
 *
 * The scalar ladiv_scalar is header-only (ladiv.h) and needs no bridge; this
 * declares only the BATCHED launcher the oracle exercises on a card, which lives
 * in device code. wwrStream_t arrives from runtime.h (a GMF cannot import); it is
 * the same type wwr.runtime_api exports, so the module forwards its stream
 * straight through. Reading the backend define that header needs is why the
 * module links wwr_backend PRIVATE -- see this directory's CMakeLists.txt.
 */

#pragma once

#include <runtime.h>

#include <cstddef>

namespace calaman::device {

/// @brief Batch @p n real-arithmetic complex divisions (a[k]+ib[k])/(c[k]+id[k])
///
/// For each k in [0, n) writes p[k] + i*q[k] = (a[k]+i*b[k]) / (c[k]+i*d[k]) by
/// Smith's algorithm (ladiv.h's ladiv_scalar). Enqueued on @p stream; returns
/// without synchronizing. Launches nothing when @p n is 0. p, q must be distinct
/// from each other; an output may alias an input.
///
/// @tparam T Real element type (float, double)
template<typename T>
void ladiv(wwr::wwrStream_t stream, std::size_t n, const T *a, const T *b, const T *c, const T *d,
           T *p, T *q);

} // namespace calaman::device
