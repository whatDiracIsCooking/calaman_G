// Tests for calaman.iterative: what stop_reason accepts and rejects, the
// IterationInfo defaults, converged(), and the EXPECT_CONVERGED message.
// Host-only; most of it is static_assert.

#include "shared/expect_converged.h"

import std;
import calaman.iterative;

namespace calaman {
namespace {

enum class Shared { Converged, MaxIterations, NumericalFailure };
enum class Extended { Converged, MaxIterations, Extra, NumericalFailure };
enum class MissingFailure { Converged, MaxIterations };
enum Unscoped { Converged, MaxIterations, NumericalFailure };
struct NotAnEnum {
  static constexpr int Converged = 0, MaxIterations = 1, NumericalFailure = 2;
};

static_assert(stop_reason<Shared>);
static_assert(stop_reason<Extended>);
static_assert(stop_reason<Unscoped>);
static_assert(!stop_reason<MissingFailure>);
static_assert(!stop_reason<NotAnEnum>);
static_assert(!stop_reason<int>);

// A method's Info, shaped like NnlsInfo.
struct DemoInfo : IterationInfo<Extended> {
  double residual = 0.0;
};

static_assert(IterationInfo<Shared>{}.iterations == 0);
static_assert(IterationInfo<Shared>{}.reason == Shared::MaxIterations);
static_assert(!converged(IterationInfo<Shared>{}));
static_assert(converged(IterationInfo<Shared>{{}, Shared::Converged}));
static_assert(noexcept(converged(IterationInfo<Shared>{})));

TEST(IterativeTests, ConvergedReadsReasonThroughADerivedInfo) {
  DemoInfo info{};
  EXPECT_FALSE(converged(info));
  info.reason = Extended::Extra;
  EXPECT_FALSE(converged(info));
  info.reason = Extended::Converged;
  info.iterations = 7;
  EXPECT_TRUE(converged(info));
  EXPECT_CONVERGED(info);
}

TEST(IterativeTests, ExpectConvergedReportsReasonAndIterations) {
  DemoInfo info{};
  info.iterations = 12;
  const auto max_it = test::is_converged(info);
  EXPECT_FALSE(max_it);
  EXPECT_NE(std::string(max_it.message()).find("reason=MaxIterations"), std::string::npos)
      << max_it.message();
  EXPECT_NE(std::string(max_it.message()).find("iterations=12"), std::string::npos)
      << max_it.message();

  info.reason = Extended::Extra;
  const auto extra = test::is_converged(info);
  EXPECT_FALSE(extra);
  EXPECT_NE(std::string(extra.message()).find("method-specific (2)"), std::string::npos)
      << extra.message();
}

} // namespace
} // namespace calaman
