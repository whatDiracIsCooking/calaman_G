/**
 * @file instantiations.cpp
 * @brief The one explicit instantiation of trttf per supported type
 *
 * Pairs with interface.cppm's `extern template` list; this list and trttf.cu's
 * must stay in step, or a type links against nothing.
 */

module calaman.trttf;

// The implicit import of the primary interface does not re-export its imports.
import std;
import wwr.runtime_api;
import wwr.complex; // wwrFloatComplex, wwrDoubleComplex

namespace calaman {

template Status trttf<float>(wwr::wwrStream_t, Trans, Uplo, std::size_t, const float *,
                             std::size_t, float *);
template Status trttf<double>(wwr::wwrStream_t, Trans, Uplo, std::size_t, const double *,
                              std::size_t, double *);
template Status trttf<wwr::wwrFloatComplex>(wwr::wwrStream_t, Trans, Uplo, std::size_t,
                                            const wwr::wwrFloatComplex *, std::size_t,
                                            wwr::wwrFloatComplex *);
template Status trttf<wwr::wwrDoubleComplex>(wwr::wwrStream_t, Trans, Uplo, std::size_t,
                                             const wwr::wwrDoubleComplex *, std::size_t,
                                             wwr::wwrDoubleComplex *);

} // namespace calaman
