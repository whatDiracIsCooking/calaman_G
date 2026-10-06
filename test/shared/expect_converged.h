// EXPECT_CONVERGED(info) -- assert an iterative method converged, and on
// failure print its reason and iteration count. info is any Info struct
// deriving from calaman::IterationInfo; converged() is found by ADL, so the
// including test only needs to import the method's module (which re-exports
// calaman.iterative). A header, not a module: a macro cannot ride an import.
#pragma once

#include <gtest/gtest.h>

namespace calaman::test {

template<class Info>
::testing::AssertionResult is_converged(const Info &info) {
  if (converged(info)) {
    return ::testing::AssertionSuccess();
  }
  using Reason = decltype(info.reason);
  const char *name = info.reason == Reason::MaxIterations      ? "MaxIterations"
                     : info.reason == Reason::NumericalFailure ? "NumericalFailure"
                                                               : "method-specific";
  return ::testing::AssertionFailure()
         << "not converged: reason=" << name << " (" << static_cast<int>(info.reason)
         << "), iterations=" << info.iterations;
}

} // namespace calaman::test

#define EXPECT_CONVERGED(info) EXPECT_TRUE(::calaman::test::is_converged(info))
