// main.cpp -- what a custom DEFAULT error policy looks like from the call site.
//
// gpu_check with no explicit policy routes a failure through
// wwr::extension::DefaultErrorPolicy<T>. In a normal build that policy prints to
// stderr and std::abort()s. Built with
//
//   -DWWR_DEFAULT_ERROR_POLICY_IMPL=<this dir>/custom_default_error_policy.h
//
// the same call instead runs custom_default_error_policy.h's handler, which logs and
// CONTINUES -- so control returns here and gpu_check yields false rather than
// aborting. Nothing in this file changes between the two builds; only which
// DefaultErrorPolicy the extension layer was compiled with does.
//
// Reaching the final line is the proof: a default build aborts before it; a
// custom-policy build prints it and exits 0. See example/README.md for how to
// run it both ways.
//
// stderr is a FILE*, which `import std;` does not provide; an ordinary #include
// beside an import is fine in a plain (non-module) translation unit like this.
#include <cstdlib>

import std;
import wwr.runtime_api;      // gpuErrorInvalidValue, gpuSuccess
import wwr.extension.common; // gpu_check, DefaultErrorPolicy

using namespace wwr;            // gpuErrorInvalidValue
using namespace wwr::extension; // gpu_check

int main() {
  std::println("triggering a GPU error through the default error policy...");

  // A deterministic, device-independent non-success code -- no driver or device
  // is needed. gpu_check only compares it against the success code and, on a
  // mismatch, hands the failure to the default policy.
  const bool ok = gpu_check(gpuErrorInvalidValue);

  std::println("gpu_check returned {} -- the custom default policy handled the error "
               "without aborting",
               ok);
  return EXIT_SUCCESS;
}
