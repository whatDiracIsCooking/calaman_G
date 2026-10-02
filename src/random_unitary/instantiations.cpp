/**
 * @file instantiations.cpp
 * @brief Explicit template instantiations for calaman.random_unitary
 *
 * The implementation unit paired with random_unitary.cppm's `extern template`
 * list: it instantiates random_unitary and random_unitary_bufferSize once, for
 * the four usual_fp types, so the symbols live here and no consumer
 * re-instantiates the bodies.
 */

module calaman.random_unitary;

// An implementation unit implicitly imports its primary interface, but an import
// is not re-exported through it -- the names in the signatures below (and the
// routines the bodies call) are not visible without these.
import std;
import wwr.solver;                   // wwrsolverStatus_t, wwrsolverDnHandle_t
import wwr.rand;                     // wwrrandState
import wwr.wrappers.common;          // usual_fp + wwrFloatComplex/wwrDoubleComplex
import wwr.extension.random_normal;  // random_normal
import calaman.orthogonalize;        // orthogonalize (+ _bufferSize)

namespace calaman {

// Function: random_unitary_bufferSize
template wwr::wwrsolverStatus_t random_unitary_bufferSize<float>(wwr::wwrsolverDnHandle_t, const int, int*);
template wwr::wwrsolverStatus_t random_unitary_bufferSize<double>(wwr::wwrsolverDnHandle_t, const int, int*);
template wwr::wwrsolverStatus_t random_unitary_bufferSize<wwr::wwrFloatComplex>(wwr::wwrsolverDnHandle_t, const int, int*);
template wwr::wwrsolverStatus_t random_unitary_bufferSize<wwr::wwrDoubleComplex>(wwr::wwrsolverDnHandle_t, const int, int*);

// Function: random_unitary
template void random_unitary<float>(wwr::wwrsolverDnHandle_t, const int, float*, float*, wwr::wwrrandState*, const int, int*, int*);
template void random_unitary<double>(wwr::wwrsolverDnHandle_t, const int, double*, double*, wwr::wwrrandState*, const int, int*, int*);
template void random_unitary<wwr::wwrFloatComplex>(wwr::wwrsolverDnHandle_t, const int, wwr::wwrFloatComplex*, wwr::wwrFloatComplex*, wwr::wwrrandState*, const int, int*, int*);
template void random_unitary<wwr::wwrDoubleComplex>(wwr::wwrsolverDnHandle_t, const int, wwr::wwrDoubleComplex*, wwr::wwrDoubleComplex*, wwr::wwrrandState*, const int, int*, int*);

} // namespace calaman
