/**
 * @file instantiations.cpp
 * @brief The one explicit instantiation of tfttp per supported type
 *
 * Pairs with interface.cppm's `extern template` list; this list and tfttp.cu's
 * must stay in step, or a type links against nothing.
 */

module calaman.tfttp;

// The implicit import of the primary interface does not re-export its imports.
import std;
import wwr.runtime_api;
import wwr.complex; // wwrFloatComplex, wwrDoubleComplex

namespace calaman {

template Status tfttp<float>(wwr::wwrStream_t, Trans, Uplo, std::size_t, const float *, float *);
template Status tfttp<double>(wwr::wwrStream_t, Trans, Uplo, std::size_t, const double *,
                              double *);
template Status tfttp<wwr::wwrFloatComplex>(wwr::wwrStream_t, Trans, Uplo, std::size_t,
                                            const wwr::wwrFloatComplex *, wwr::wwrFloatComplex *);
template Status tfttp<wwr::wwrDoubleComplex>(wwr::wwrStream_t, Trans, Uplo, std::size_t,
                                             const wwr::wwrDoubleComplex *,
                                             wwr::wwrDoubleComplex *);

} // namespace calaman
