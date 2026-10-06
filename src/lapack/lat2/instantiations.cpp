/**
 * @file instantiations.cpp
 * @brief The explicit instantiations of lat2() per (From, To) pair
 *
 * Implementation unit of calaman.lat2, paired with interface.cppm's `extern
 * template` list: the wrapper names the GMF-only device::lat2, so it is
 * instantiated once here. Keep this list in step with lat2.cu's.
 */

module calaman.lat2;

// An implementation unit's implicit import of its interface does not re-export
// that interface's imports; the signatures below need these.
import std;
import wwr.runtime_api;
import wwr.complex;
import calaman.common; // Uplo

namespace calaman {

template Status lat2<double, float>(wwr::wwrStream_t, Uplo, std::size_t, const double *,
                                    std::size_t, float *, std::size_t, int *);
template Status lat2<wwr::wwrDoubleComplex, wwr::wwrFloatComplex>(
    wwr::wwrStream_t, Uplo, std::size_t, const wwr::wwrDoubleComplex *, std::size_t,
    wwr::wwrFloatComplex *, std::size_t, int *);

} // namespace calaman
