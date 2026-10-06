/**
 * @file instantiations.cpp
 * @brief The one explicit instantiation of tpttf per supported type
 *
 * Pairs with interface.cppm's `extern template` list; this list and tpttf.cu's
 * must stay in step, or a type links against nothing.
 */

module calaman.tpttf;

// The implicit import of the primary interface does not re-export its imports.
import std;
import wwr.runtime_api;
import wwr.complex; // wwrFloatComplex, wwrDoubleComplex

namespace calaman {

template Status tpttf<float>(wwr::wwrStream_t, Trans, Uplo, std::size_t, const float *, float *);
template Status tpttf<double>(wwr::wwrStream_t, Trans, Uplo, std::size_t, const double *,
                              double *);
template Status tpttf<wwr::wwrFloatComplex>(wwr::wwrStream_t, Trans, Uplo, std::size_t,
                                            const wwr::wwrFloatComplex *, wwr::wwrFloatComplex *);
template Status tpttf<wwr::wwrDoubleComplex>(wwr::wwrStream_t, Trans, Uplo, std::size_t,
                                             const wwr::wwrDoubleComplex *,
                                             wwr::wwrDoubleComplex *);

} // namespace calaman
