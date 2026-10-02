/**
 * @file instantiations.cpp
 * @brief The one explicit instantiation of laset per supported type
 *
 * Implementation unit of calaman.laset. Pairs with the `extern template`
 * declarations in interface.cppm: together they keep every importer from
 * instantiating the wrapper again -- its body names the device launcher
 * declared only in the interface's global module fragment, so it must be
 * instantiated here, inside this library. The device-side work it calls is
 * instantiated separately, in laset.cu, as device code.
 *
 * This list and the .cu's must stay in step -- a type added here without being
 * added there links against nothing.
 */

module calaman.laset;

// An implementation unit implicitly imports its primary interface, but an
// import is not re-exported through it -- the names in the signatures below are
// not visible without these. (Region is re-exported by the interface, so the
// implicit import would carry it; it is named here anyway for the same
// signatures-name-their-modules reason.)
import std;
import wwr.runtime_api;
import calaman.common; // Region

namespace calaman {

template void laset<float>(wwr::wwrStream_t, Region, std::size_t, std::size_t, float, float,
                           float *, std::size_t);
template void laset<double>(wwr::wwrStream_t, Region, std::size_t, std::size_t, double, double,
                            double *, std::size_t);

} // namespace calaman
