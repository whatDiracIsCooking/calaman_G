// main.cpp -- what using an installed gpumod actually looks like.
//
// This consumes ONLY the parts of gpumod that the package installs: the
// backend-neutral gpu* layer (wwr.runtime_api, wwr.blas) and the wrappers
// (wwr.wrappers.* -- the dispatch wrappers, plus the untyped tools-extension
// wrapper wwr.wrappers.tx). It deliberately does NOT touch
// the extension layer (wwr.extension.*, the RAII handle / buffer / error
// abstractions) -- that layer is built in-tree but is not part of the installed
// package, so a find_package consumer cannot see it. Restoring it to the
// package is a separate decision (a WWR_BUILD_EXTENSION opt-in); until then
// an installed consumer manages its own device memory and handles, exactly as
// this file does.
//
// Two things are proved, and they fail differently:
//
//   * COMPILE + LINK proves the install. `import` resolving means the module
//     SOURCES were installed and the export set re-attaches them (a C++23
//     module package ships sources -- a BMI is not portable -- and this build
//     compiles them). Each wrapper's units #include "wrappers/.../dispatch_*.h"
//     from their global module fragment, so those headers had to travel next to
//     the sources; wwr.wrappers.sparse also needs the WWR_GPU_BACKEND_*
//     define at install-compile time. Taking the address of one instantiation
//     per wrapper (solver/fft/sparse below) forces each to resolve and link
//     without running a kernel -- so this half needs no GPU.
//   * RUN proves it works. multiply_square does a real gemm on the device:
//     allocate, copy up, dispatch gemm<float,int> (cublasSgemm on CUDA,
//     hipblasSgemm on HIP -- named nowhere here, which is the point), copy down,
//     check. This half needs a device; with none it reports 77 (ctest's skip)
//     and the compile-and-link result still stands.

// stderr is a FILE*, which `import std;` does not give you -- the std module
// exports the std:: names, not the C library's macros and objects. An ordinary
// #include beside an import is fine in a plain translation unit like this one;
// it is only inside a MODULE unit's global module fragment that this project's
// headers and `import std;` would collide.
#include <cstdio>

import std;

import wwr.runtime_api; // gpuMalloc, gpuMemcpy, gpuGetDevice, gpuSuccess
import wwr.blas;        // gpublasHandle_t, gpublasCreate, GPUBLAS_OP_N
import wwr.wrappers.blas;
import wwr.wrappers.solver;
import wwr.wrappers.fft;
import wwr.wrappers.sparse;
import wwr.wrappers.tx; // wwr::tx::mark, ScopedRange, range_start/stop

// The gpu* names (gpuSuccess, gpuMalloc, GPUBLAS_OP_N, ...) and the wrappers
// (gemm, potri, ...) are all exported in namespace wwr. A consumer is not
// inside it, so unlike this project's own tests it has to say so.
using namespace wwr;

namespace {

// C = A * B for square column-major matrices, via the backend's BLAS.
//
// A is the identity and B is 1, 2, 3, ..., so A * B == B and the check needs no
// tolerance. Device memory and the handle are managed by hand here -- the RAII
// wrappers that would do this live in the (uninstalled) extension layer.
bool multiply_square(const int n) {
  const std::size_t count = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);

  std::vector<float> host_a(count, 0.0f);
  std::vector<float> host_b(count);
  for (std::size_t i = 0; i < count; ++i) {
    host_b[i] = static_cast<float>(i + 1);
  }
  for (int i = 0; i < n; ++i) {
    host_a[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
           static_cast<std::size_t>(i)] = 1.0f;
  }

  const std::size_t bytes = count * sizeof(float);
  float *a = nullptr;
  float *b = nullptr;
  float *c = nullptr;
  if (gpuMalloc(reinterpret_cast<void **>(&a), bytes) != gpuSuccess ||
      gpuMalloc(reinterpret_cast<void **>(&b), bytes) != gpuSuccess ||
      gpuMalloc(reinterpret_cast<void **>(&c), bytes) != gpuSuccess) {
    std::println(stderr, "device allocation failed");
    return false;
  }

  // One cleanup path for every early return below. gpuFree's status is
  // discarded on purpose -- this is best-effort teardown -- and the casts are
  // load-bearing: hipFree is [[nodiscard]] where cudaFree is not, so without
  // them the example warns under HIP and is clean under CUDA.
  const auto teardown = [&] {
    (void)gpuFree(a);
    (void)gpuFree(b);
    (void)gpuFree(c);
  };

  if (gpuMemcpy(a, host_a.data(), bytes, gpuMemcpyHostToDevice) != gpuSuccess ||
      gpuMemcpy(b, host_b.data(), bytes, gpuMemcpyHostToDevice) != gpuSuccess) {
    std::println(stderr, "host -> device copy failed");
    teardown();
    return false;
  }

  gpublasHandle_t handle{};
  if (gpublasCreate(&handle) != GPUBLAS_STATUS_SUCCESS) {
    std::println(stderr, "gpublasCreate failed");
    teardown();
    return false;
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  const auto status = gemm<float, int>(handle, GPUBLAS_OP_N, GPUBLAS_OP_N, n, n, n, &alpha, a, n, b,
                                       n, &beta, c, n);
  gpublasDestroy(handle);
  if (status != GPUBLAS_STATUS_SUCCESS) {
    std::println(stderr, "gemm failed with status {}", static_cast<int>(status));
    teardown();
    return false;
  }

  std::vector<float> host_c(count);
  const bool copied = gpuMemcpy(host_c.data(), c, bytes, gpuMemcpyDeviceToHost) == gpuSuccess &&
                      gpuDeviceSynchronize() == gpuSuccess;
  teardown();
  if (!copied) {
    std::println(stderr, "device -> host copy failed");
    return false;
  }

  for (std::size_t i = 0; i < count; ++i) {
    if (host_c[i] != host_b[i]) {
      std::println(stderr, "gemm mismatch at {}: got {}, want {}", i, host_c[i], host_b[i]);
      return false;
    }
  }

  std::println("gemm   : {0}x{0} identity * B == B, {1} elements checked", n, count);
  return true;
}

// solver / fft / sparse are proved at compile and link time only: taking the
// address of one instantiation each forces its wrapper to resolve and its
// archive to link, without needing a device. The volatile sink keeps the
// compiler from folding the reads away.
bool wrappers_link() {
  static const void *volatile sink[] = {
      reinterpret_cast<const void *>(&potri<float>),    // wwr.wrappers.solver
      reinterpret_cast<const void *>(&exec_c2c<float>), // wwr.wrappers.fft
      reinterpret_cast<const void *>(&bsrmv<float>),    // wwr.wrappers.sparse
  };
  for (const void *volatile p : sink) {
    if (p == nullptr)
      return false;
  }
  std::println("link   : solver/fft/sparse wrappers resolved and linked");
  return true;
}

} // namespace

int main() {
  // wwr.wrappers.tx is host-side profiler annotation -- it needs no device,
  // so exercising the whole surface here (marker, RAII push/pop via ScopedRange,
  // async start/stop) proves the tx wrapper chain -- wwr.wrappers.tx ->
  // wwr.tx -> the backend's NVTX/rocTX module -- installs, compiles, links
  // AND runs, regardless of whether a GPU is present. The ScopedRange guards the
  // rest of main, so it pops on every return path below.
  tx::mark("consumer start");
  const tx::ScopedRange session{"gpumod install-check"};
  tx::range_stop(tx::range_start("async probe"));
  std::println("tx     : marker + scoped/async ranges resolved and linked");

  // Compile-and-link proof first: no device needed, and it is what proves the
  // install regardless of whether a GPU is present to run the gemm.
  if (!wrappers_link())
    return 1;

  int device = 0;
  if (gpuGetDevice(&device) != gpuSuccess) {
    std::println("gpumod : installed package consumed and linked; no GPU to run the gemm");
    return 77; // ctest's conventional "skipped"
  }

  if (!multiply_square(64))
    return 1;

  std::println("gpumod : consumed from an installed package, all checks passed");
  return 0;
}
