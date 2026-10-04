/**
 * @file interface.cppm
 * @brief Primary interface for calaman.lartg -- generate plane rotations, the
 *        batched device lift of LAPACK's ?lartg
 *
 * One device routine: given length-@p n arrays @p f and @p g, write the n plane
 * rotations that each take (f[k], g[k]) to (r[k], 0) -- the cosine @p c, sine
 * @p s, and rotated length @p r. Reference ?lartg is a SCALAR routine, nothing
 * to offload alone, so this batches n independent ones (the shape a bdsqr-style
 * diagonal sweep wants). Enqueued on the given stream and returns WITHOUT
 * synchronizing, like a BLAS call; the caller synchronizes when it needs the
 * outputs. All arrays are device pointers the caller owns; nothing is allocated
 * here, so a stream -- not a device handle -- is the whole requirement.
 *
 * REAL ONLY. ?lartg's complex variants (clartg / zlartg) carry a real cosine but
 * a complex sine and a materially different algorithm -- the wall calaman.larfg
 * documents -- so the surface is float and double. Mapping from ?lartg
 * (docs/architecture.md §4): the scalars (F,G,C,S,R) become per-element arrays,
 * @p n is std::size_t, and there is no INFO (this returns calaman::Status
 * carrying only a kernel-launch error).
 *
 * `extern template` below pairs with instantiations.cpp: the wrapper is
 * instantiated once inside this library, so an importer never re-instantiates a
 * body that names the .cu-side launcher declared only in the GMF.
 *
 * Usage:
 *   import calaman.lartg;     // also re-exports calaman::Status
 *   import wwr.runtime_api;   // wwrStream_t, wwrStreamCreate
 *   wwr::wwrStream_t stream{};
 *   wwr::wwrStreamCreate(&stream);
 *   // d_f, d_g: length-n device inputs; d_c, d_s, d_r: length-n device outputs
 *   calaman::lartg(stream, n, d_f, d_g, d_c, d_s, d_r);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the export import
// below supplies.
#include "error_handling/error_macros.h"

#include "lartg_bridge.h"

export module calaman.lartg;

import std;
import wwr.runtime_api;

// export import, not a plain import: lartg RETURNS calaman::Status, so a
// consumer of `import calaman.lartg;` must see Status's member functions, not
// just its name -- the same re-export lacgv / lascl2 do.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

// Not an `export namespace` block: an explicit instantiation declaration
// (`extern template`) cannot be exported, so the template carries its own
// `export` and the declarations below sit in the plain namespace.

/// @brief Generate @p n plane rotations from @p f, @p g on @p stream (?lartg)
///
/// For each k in [0, n) writes (c[k], s[k], r[k]) so that
/// [c -s; s c] * [f[k]; g[k]] = [r[k]; 0], by reference ?lartg's safe-scaled
/// formula (c >= 0; r = sign(f) * hypot(f, g)). Enqueues nothing and returns
/// success when @p n is 0. The three output arrays must be distinct from one
/// another; an output may alias an input. All arrays live on @p stream's device.
///
/// @tparam T Element type; one of the instantiated types (float, double)
/// @param stream Stream the launch is enqueued on; the arrays live on its device
/// @param n Number of rotations (length of every array)
/// @param f Device input, the first components
/// @param g Device input, the second components
/// @param c Device output, the cosines
/// @param s Device output, the sines
/// @param r Device output, the rotated lengths
/// @return Success, or the runtime error the kernel launch reported
export template<typename T>
Status lartg(const wwr::wwrStream_t stream, const std::size_t n, const T *f, const T *g, T *c, T *s,
             T *r) {
  if (n == 0) {
    return wwr::wwrSuccess;
  }
  device::lartg(stream, n, f, g, c, s, r);
  // The launcher returns void, so the only way to catch a bad launch is the
  // runtime's sticky error -- checked the moment it is enqueued, as lacgv does.
  CLM_TRY(wwr::wwrGetLastError());
  return wwr::wwrSuccess;
}

extern template Status lartg<float>(wwr::wwrStream_t, std::size_t, const float *, const float *,
                                    float *, float *, float *);
extern template Status lartg<double>(wwr::wwrStream_t, std::size_t, const double *, const double *,
                                     double *, double *, double *);

} // namespace calaman
