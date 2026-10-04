/**
 * @file instantiations.cpp
 * @brief The one explicit instantiation of orghr per supported type
 *
 * Implementation unit of calaman.orghr. Pairs with the `extern template`
 * declarations in interface.cppm: together they keep every importer from
 * instantiating the driver again -- its body names the device launcher declared
 * only in the interface's global module fragment, so it must be instantiated
 * here, inside this library. The device-side work it calls is instantiated
 * separately, in orghr.cu, as device code.
 *
 * This list and the .cu's must stay in step -- a type added here without being
 * added there links against nothing. orghr_bufferSize is a header-only template
 * (no device launcher in its body), so it needs no entry here.
 */

module calaman.orghr;

// An implementation unit implicitly imports its primary interface, but an import
// is not re-exported through it -- the names in the signatures below are not
// visible without these.
import std;
import wwr.solver; // wwrsolverDnHandle_t

namespace calaman {

template Status orghr<float>(wwr::wwrsolverDnHandle_t, int, int, int, float *, int, const float *,
                             void *, std::size_t, int *);
template Status orghr<double>(wwr::wwrsolverDnHandle_t, int, int, int, double *, int,
                              const double *, void *, std::size_t, int *);

} // namespace calaman
