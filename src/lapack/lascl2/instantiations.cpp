/**
 * @file instantiations.cpp
 * @brief The one explicit instantiation of lascl2 per supported type
 *
 * Implementation unit of calaman.lascl2. Pairs with the `extern template`
 * declarations in interface.cppm: together they keep every importer from
 * instantiating the wrapper again -- its body names the device launcher
 * declared only in the interface's global module fragment, so it must be
 * instantiated here, inside this library. The device-side work it calls is
 * instantiated separately, in lascl2.cu, as device code.
 *
 * This list and the .cu's must stay in step -- a type added here without being
 * added there links against nothing.
 */

module calaman.lascl2;

// An implementation unit implicitly imports its primary interface, but an
// import is not re-exported through it -- the names in the signatures below are
// not visible without these.
import std;
import wwr.runtime_api;

namespace calaman {

template Status lascl2<float>(wwr::wwrStream_t, std::size_t, std::size_t, const float *, float *,
                              std::size_t);
template Status lascl2<double>(wwr::wwrStream_t, std::size_t, std::size_t, const double *, double *,
                               std::size_t);

} // namespace calaman
