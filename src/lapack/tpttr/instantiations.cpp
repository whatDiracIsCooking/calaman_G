/**
 * @file instantiations.cpp
 * @brief The one explicit instantiation of tpttr per supported type
 *
 * Pairs with interface.cppm's `extern template` list; this list and tpttr.cu's
 * must stay in step, or a type links against nothing.
 */

module calaman.tpttr;

// The implicit import of the primary interface does not re-export its imports.
import std;
import wwr.runtime_api;
import wwr.complex; // wwrFloatComplex, wwrDoubleComplex

namespace calaman {

template Status tpttr<float>(wwr::wwrStream_t, Uplo, std::size_t, const float *, float *,
                             std::size_t);
template Status tpttr<double>(wwr::wwrStream_t, Uplo, std::size_t, const double *, double *,
                              std::size_t);
template Status tpttr<wwr::wwrFloatComplex>(wwr::wwrStream_t, Uplo, std::size_t,
                                            const wwr::wwrFloatComplex *, wwr::wwrFloatComplex *,
                                            std::size_t);
template Status tpttr<wwr::wwrDoubleComplex>(wwr::wwrStream_t, Uplo, std::size_t,
                                             const wwr::wwrDoubleComplex *,
                                             wwr::wwrDoubleComplex *, std::size_t);

} // namespace calaman
