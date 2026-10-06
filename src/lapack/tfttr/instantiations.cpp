/**
 * @file instantiations.cpp
 * @brief The one explicit instantiation of tfttr per supported type
 *
 * Pairs with interface.cppm's `extern template` list; this list and tfttr.cu's
 * must stay in step, or a type links against nothing.
 */

module calaman.tfttr;

// The implicit import of the primary interface does not re-export its imports.
import std;
import wwr.runtime_api;
import wwr.complex; // wwrFloatComplex, wwrDoubleComplex

namespace calaman {

template Status tfttr<float>(wwr::wwrStream_t, Trans, Uplo, std::size_t, const float *, float *,
                             std::size_t);
template Status tfttr<double>(wwr::wwrStream_t, Trans, Uplo, std::size_t, const double *,
                              double *, std::size_t);
template Status tfttr<wwr::wwrFloatComplex>(wwr::wwrStream_t, Trans, Uplo, std::size_t,
                                            const wwr::wwrFloatComplex *, wwr::wwrFloatComplex *,
                                            std::size_t);
template Status tfttr<wwr::wwrDoubleComplex>(wwr::wwrStream_t, Trans, Uplo, std::size_t,
                                             const wwr::wwrDoubleComplex *,
                                             wwr::wwrDoubleComplex *, std::size_t);

} // namespace calaman
