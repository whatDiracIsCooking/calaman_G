/**
 * @file instantiations.cpp
 * @brief Explicit template instantiations for calaman.orthogonalize
 *
 * The implementation unit paired with orthogonalize.cppm's `extern template`
 * list: it instantiates orthogonalize and orthogonalize_bufferSize once, for the
 * four usual_fp types, so the symbols live here and no consumer re-instantiates
 * the bodies.
 */

module calaman.orthogonalize;

// An implementation unit implicitly imports its primary interface, but an import
// is not re-exported through it -- the names in the signatures below (and the
// wrappers the bodies call) are not visible without these.
import std;
import wwr.solver;            // wwrsolverStatus_t, wwrsolverDnHandle_t, WWRSOLVER_STATUS_*
import wwr.wrappers.solver;   // geqrf/orgqr/ungqr (+ bufferSize) + wwrFloatComplex/wwrDoubleComplex
import calaman.common;        // align_up

namespace calaman {

// Function: orthogonalize_bufferSize
template Status orthogonalize_bufferSize<float>(wwr::wwrsolverDnHandle_t, const int, const int, int*);
template Status orthogonalize_bufferSize<double>(wwr::wwrsolverDnHandle_t, const int, const int, int*);
template Status orthogonalize_bufferSize<wwr::wwrFloatComplex>(wwr::wwrsolverDnHandle_t, const int, const int, int*);
template Status orthogonalize_bufferSize<wwr::wwrDoubleComplex>(wwr::wwrsolverDnHandle_t, const int, const int, int*);

// Function: orthogonalize
template Status orthogonalize<float>(wwr::wwrsolverDnHandle_t, const int, const int, float*, float*, const int, int*, int*);
template Status orthogonalize<double>(wwr::wwrsolverDnHandle_t, const int, const int, double*, double*, const int, int*, int*);
template Status orthogonalize<wwr::wwrFloatComplex>(wwr::wwrsolverDnHandle_t, const int, const int, wwr::wwrFloatComplex*, wwr::wwrFloatComplex*, const int, int*, int*);
template Status orthogonalize<wwr::wwrDoubleComplex>(wwr::wwrsolverDnHandle_t, const int, const int, wwr::wwrDoubleComplex*, wwr::wwrDoubleComplex*, const int, int*, int*);

} // namespace calaman
