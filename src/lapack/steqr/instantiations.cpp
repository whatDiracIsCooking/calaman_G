/**
 * @file instantiations.cpp
 * @brief The one explicit instantiation of steqr / steqr_bufferSize per type
 *
 * Implementation unit of calaman.steqr, paired with interface.cppm's `extern
 * template` list: the wrapper names the .cu-side launcher declared only in the
 * interface's GMF, so it is instantiated here, inside this library. Keep this
 * list, the interface's and steqr.cu's in step (s/d/c/z).
 */

module calaman.steqr;

// An implementation unit's implicit import of its interface does not re-export
// that interface's imports; the signatures below need these.
import std;
import wwr.runtime_api;
import wwr.complex;
import calaman.common; // CompZ

namespace calaman {

template std::size_t steqr_bufferSize<float>(CompZ, int);
template std::size_t steqr_bufferSize<double>(CompZ, int);
template std::size_t steqr_bufferSize<wwr::wwrFloatComplex>(CompZ, int);
template std::size_t steqr_bufferSize<wwr::wwrDoubleComplex>(CompZ, int);
template Status steqr<float>(wwr::wwrStream_t, CompZ, int, float *, float *, float *, int, void *,
                             std::size_t, int *);
template Status steqr<double>(wwr::wwrStream_t, CompZ, int, double *, double *, double *, int,
                              void *, std::size_t, int *);
template Status steqr<wwr::wwrFloatComplex>(wwr::wwrStream_t, CompZ, int, float *, float *,
                                            wwr::wwrFloatComplex *, int, void *, std::size_t,
                                            int *);
template Status steqr<wwr::wwrDoubleComplex>(wwr::wwrStream_t, CompZ, int, double *, double *,
                                             wwr::wwrDoubleComplex *, int, void *, std::size_t,
                                             int *);

} // namespace calaman
