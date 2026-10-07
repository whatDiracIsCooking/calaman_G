// Suite for calaman.error_handling -- the Status value type that unifies the
// three WarpWraps error domains (wwrblasStatus_t, wwrsolverStatus_t, wwrError_t)
// behind one return type.
//
// Host-only ON PURPOSE: Status is pure host control flow -- it names enum
// constants and forwards to the vendor string tables, never touching the device.
// So this is NOT labeled REQUIRES_GPU and runs on a card-less CI runner, unlike
// the numerical suites. It needs no reference-LAPACK oracle either, so there is
// no calaman::lapack_reference guard in its CMakeLists.
//
// The headline is CrossDomainErrorKeepsItsDomain: a runtime failure carried as a
// Status reports ErrorDomain::runtime, not a stand-in BLAS code -- the lossy
// masquerade in calaman.diff_norm's ell_inf path is exactly what Status removes.

#include <gtest/gtest.h>

import std;

import calaman.error_handling;
import wwr.extension.common; // error_type, gpu_check, success_code
import wwr.blas;             // WWRBLAS_STATUS_SUCCESS / WWRBLAS_STATUS_NOT_INITIALIZED
import wwr.solver;           // WWRSOLVER_STATUS_SUCCESS
import wwr.runtime_api;      // wwrSuccess / wwrErrorInvalidValue

namespace calaman {
namespace {

// A non-empty string is the bar for name()/message(): the exact spelling is the
// vendor's (cublas vs hipblas differ), so the suite checks routing, not text.
bool nonempty(const char *s) {
  return s != nullptr && std::string_view{s}.size() > 0;
}

// --- constexpr guarantee ----------------------------------------------------
// The converting constructors are constexpr, and ok() is constexpr-evaluable for
// the blas/solver domains -- their success_code<T>() is constexpr upstream. The
// runtime domain's success_code<wwrError_t>() is NOT constexpr in WarpWraps, so
// ok() on a runtime Status cannot appear in a constant expression (that path is
// covered by the runtime EXPECTs below); only its domain, set in the constexpr
// constructor, is a constant. If any of this regressed, these fail to compile.
static_assert(Status{wwr::WWRBLAS_STATUS_SUCCESS}.ok());
static_assert(Status{wwr::WWRSOLVER_STATUS_SUCCESS}.ok());
static_assert(!Status{wwr::WWRBLAS_STATUS_NOT_INITIALIZED}.ok());
static_assert(Status{wwr::wwrSuccess}.domain == ErrorDomain::runtime);

// --- Status is a registered error_type --------------------------------------
// The whole point of the three wwr::extension specialisations: Status satisfies
// the concept, so it is accepted wherever an error_type is required (gpu_check,
// error policies). success_code<Status>() is a success, and operator== is
// outcome-based, so a success from ANY domain equals it and an error does not.
static_assert(wwr::extension::error_type<Status>);
static_assert(wwr::extension::success_code<Status>().ok());
static_assert(Status{wwr::WWRBLAS_STATUS_SUCCESS} == Status{wwr::WWRSOLVER_STATUS_SUCCESS});
static_assert(Status{wwr::WWRBLAS_STATUS_NOT_INITIALIZED} != Status{wwr::WWRBLAS_STATUS_SUCCESS});

// --- each domain casts in, tagged, and judges success by its own constant -----

TEST(ErrorHandlingTests, BlasStatusCastsInTaggedBlas) {
  const Status ok = wwr::WWRBLAS_STATUS_SUCCESS;
  EXPECT_EQ(ok.domain, ErrorDomain::blas);
  EXPECT_TRUE(ok.ok());

  const Status bad = wwr::WWRBLAS_STATUS_NOT_INITIALIZED;
  EXPECT_EQ(bad.domain, ErrorDomain::blas);
  EXPECT_FALSE(bad.ok());
  EXPECT_TRUE(nonempty(bad.name()));
  EXPECT_TRUE(nonempty(bad.message()));
}

TEST(ErrorHandlingTests, SolverStatusCastsInTaggedSolver) {
  const Status ok = wwr::WWRSOLVER_STATUS_SUCCESS;
  EXPECT_EQ(ok.domain, ErrorDomain::solver);
  EXPECT_TRUE(ok.ok());

  // WarpWraps exposes only WWRSOLVER_STATUS_SUCCESS as a portable constant; the
  // non-success enumerators have no neutral WWR_* name. ok() compares against
  // success_code, so any non-success value suffices to exercise the failure path.
  const Status bad = static_cast<wwr::wwrsolverStatus_t>(1);
  EXPECT_EQ(bad.domain, ErrorDomain::solver);
  EXPECT_FALSE(bad.ok());
  EXPECT_TRUE(nonempty(bad.name()));
  EXPECT_TRUE(nonempty(bad.message()));
}

TEST(ErrorHandlingTests, RuntimeErrorCastsInTaggedRuntime) {
  const Status ok = wwr::wwrSuccess;
  EXPECT_EQ(ok.domain, ErrorDomain::runtime);
  EXPECT_TRUE(ok.ok());

  const Status bad = wwr::wwrErrorInvalidValue;
  EXPECT_EQ(bad.domain, ErrorDomain::runtime);
  EXPECT_FALSE(bad.ok());
  EXPECT_TRUE(nonempty(bad.name()));
  EXPECT_TRUE(nonempty(bad.message()));
}

// --- the headline: no cross-domain masquerade -------------------------------
// diff_norm's ell_inf path once returned WWRBLAS_STATUS_NOT_INITIALIZED for a
// failed device copy -- a BLAS code for a runtime error. Carried as a Status, a
// runtime failure keeps ErrorDomain::runtime and renders with the runtime
// string tables, never reading as the BLAS code it is not.
TEST(ErrorHandlingTests, CrossDomainErrorKeepsItsDomain) {
  const Status runtime_fail = wwr::wwrErrorInvalidValue;
  const Status blas_fail = wwr::WWRBLAS_STATUS_NOT_INITIALIZED;

  EXPECT_EQ(runtime_fail.domain, ErrorDomain::runtime);
  EXPECT_NE(runtime_fail.domain, blas_fail.domain);
  // Different domains, different name tables -- the runtime error is not spelled
  // as the BLAS one.
  EXPECT_NE(std::string_view{runtime_fail.name()}, std::string_view{blas_fail.name()});
}

// --- implicit conversion is what makes `return <raw status>;` work -----------
Status returns_blas_success() {
  return wwr::WWRBLAS_STATUS_SUCCESS;
}
Status returns_runtime_failure() {
  return wwr::wwrErrorInvalidValue;
}

TEST(ErrorHandlingTests, ImplicitConversionOnReturn) {
  EXPECT_TRUE(returns_blas_success().ok());
  EXPECT_EQ(returns_blas_success().domain, ErrorDomain::blas);

  const Status s = returns_runtime_failure();
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.domain, ErrorDomain::runtime);
}

// --- Status drives WarpWraps's gpu_check + error_policy unchanged ------------
// A minimal error_policy<_, Status>: it records how many failures it was handed,
// so the test can see that gpu_check routes exactly the non-success Statuses to
// it and leaves successes alone -- the behaviour the wwr* enums get for free, now
// available to a cross-domain Status.
struct RecordingPolicy {
  using error_type = Status;
  int *handled;
  void handle_error(Status, std::source_location) noexcept { ++*handled; }
};

TEST(ErrorHandlingTests, DrivesGpuCheckAndPolicy) {
  int handled = 0;

  // A success in ANY domain: gpu_check returns true and never calls the policy.
  EXPECT_TRUE(wwr::extension::gpu_check(Status{wwr::wwrSuccess}, RecordingPolicy{&handled}));
  EXPECT_TRUE(
      wwr::extension::gpu_check(Status{wwr::WWRBLAS_STATUS_SUCCESS}, RecordingPolicy{&handled}));
  EXPECT_EQ(handled, 0);

  // A runtime failure is detected and routed to the policy -- one handled error.
  EXPECT_FALSE(
      wwr::extension::gpu_check(Status{wwr::wwrErrorInvalidValue}, RecordingPolicy{&handled}));
  EXPECT_EQ(handled, 1);
}

// --- devinfo_verdict: the host-side verdict on a device solver's devInfo -------
// Zero is a success; any nonzero devInfo (a negative bad-argument index or a
// positive numerical failure) is WWRBLAS_STATUS_INTERNAL_ERROR in the BLAS domain.
static_assert(devinfo_verdict(0).ok());
static_assert(devinfo_verdict(1) == Status{wwr::WWRBLAS_STATUS_INTERNAL_ERROR});

TEST(ErrorHandlingTests, DevinfoVerdictZeroIsSuccess) {
  EXPECT_TRUE(devinfo_verdict(0).ok());
}

TEST(ErrorHandlingTests, DevinfoVerdictNonzeroIsBlasInternalError) {
  for (const int info : {1, 7, -1, -4}) {
    const Status verdict = devinfo_verdict(info);
    EXPECT_FALSE(verdict.ok()) << "info=" << info;
    EXPECT_EQ(verdict.domain, ErrorDomain::blas) << "info=" << info;
    EXPECT_EQ(verdict.code, static_cast<int>(wwr::WWRBLAS_STATUS_INTERNAL_ERROR))
        << "info=" << info;
  }
}

} // namespace
} // namespace calaman
