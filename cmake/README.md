# cmake/ — Build Utilities

CMake modules for C++23 module libraries and testing.

**These macros arrived with the scaffolding and have no call sites yet** — `src/`
and `test/` are empty. They are known-good in gpumod, which is where they came
from and where they are exercised (vendored at `deps/gpumod/cmake/`); read
anything here as "the contract these macros offer", not as "what this tree
already does". Two carry assumptions that are gpumod's and not this project's,
flagged in place below: `calaman_install.cmake` and `calamanConfig.cmake.in`.

## Files

```
cmake/
├── README.md                                   # This file
├── calaman_add_gpu_device_library.cmake        # Macro: backend-neutral device-kernel static libraries
├── calaman_add_cxx_module_library.cmake        # Macro: C++23 module libraries
├── calaman_add_gtest_executable.cmake          # Consolidates GoogleTest executable boilerplate
├── calaman_add_gtest_suite_tests.cmake         # Registers a GoogleTest binary with ctest, one entry per suite
├── calaman_add_interface_library.cmake         # Macro: INTERFACE libraries
├── calaman_add_test_executable.cmake           # Plain (non-GoogleTest) ctest-registered executables
├── calaman_check_gtest_suites.cmake            # Script mode: fails when a suite in the binary is missing from the CMake list
├── calaman_install.cmake                       # Install rules, the export set and the CMake package
├── calamanConfig.cmake.in                      # Template for the installed calamanConfig.cmake
└── calaman_internal_helpers.cmake              # Internal helpers (alias creation, include dirs, linking)
```

gpumod additionally has a `wwr_add_dispatch_check.cmake` — a build-time check
that a wrapper's instantiations really call the vendor entry points its TOML table
names, read back out of the compiled objects with `llvm-objdump`. **It was
deliberately not carried over**, because it needs a checker script
(`test/shared/dispatch.py` there) and dispatch tables that do not exist here yet.
Copy it from `deps/gpumod/` when this project grows its own `s/d/c/z` dispatch —
it is exactly the check that catches a `getrf` template silently calling the
single-precision entry point for `double`.

---

## Build Directory Layout

After configuring and building, the `build/` directory has this structure:

```
build/
├── _deps/                  # CMake FetchContent -- GoogleTest
│   ├── googletest-build/
│   ├── googletest-src/
│   └── googletest-subbuild/
├── deps/                   # deps/CMakeLists.txt, added via add_subdirectory
│   └── gpumod/             # the submodule's own build tree, one dir per module
├── lib/                    # Static libraries (gtest, gmock)
├── src/                    # Per-module build artifacts. ONE tree, not one per
│                           # backend: the backend split is gpumod's job
└── test/                   # Test executables
```

Every configure preset has its own `binaryDir`, so the tree above is `build/`
for the `default` preset, `build-hip/` for `hip`, and so on — they must stay
separate, or two presets sharing one directory silently reconfigure it back and
forth, a full rebuild each way.

`_deps/` is managed by CMake's FetchContent — its name is a CMake convention and should not be changed. `deps/` holds local dependencies that live in the source tree under `deps/`.

---

## `calaman_add_gpu_device_library`

Creates the STATIC library that holds a module's device-kernel `.cu` sources, compiled for **whichever backend the build selected**. Every module that owns a kernel declares one next to its `calaman_add_cxx_module_library()` call.

A `.cu` in this tree means "device pass", not "nvcc" — the same sense as the `.cuh` headers it includes. Compiling one needs a language decision and, under HIP, two driver flags; a C++23 module interface unit must get neither, being an ordinary host CXX compile. Keeping the device sources in their own library is what keeps the two sets of flags apart.

### Usage

```cmake
calaman_add_gpu_device_library(
  NAME library_name
  SOURCES file1.cu [file2.cu ...]
  [LINK_PRIVATE lib1 lib2 ...]
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `NAME` | required | Target name, by convention the owning module's name plus `.device` |
| `SOURCES` | required | One or more `.cu` sources |
| `LINK_PRIVATE` | optional | Private link dependencies |

### Always applied

- `add_library(${NAME} STATIC ${SOURCES})`
- **CUDA backend:** `CUDA_STANDARD 20`, `CUDA_STANDARD_REQUIRED ON`, `CUDA_SEPARABLE_COMPILATION OFF`, `CUDA_RESOLVE_DEVICE_SYMBOLS OFF`, `POSITION_INDEPENDENT_CODE ON`
- **HIP backend:** `LANGUAGE CXX` on every source (a HIP build enables no CUDA language, and `hip::device`'s `-x hip` is gated on `$<COMPILE_LANGUAGE:CXX>`), `hip::device` linked PRIVATE, and `-B<rocm>/llvm/bin` so the `amdgcn-link` step finds ROCm's `lld`

### What it deliberately does not take

No `LINK_PUBLIC`, no `INCLUDE_DIRS_*`, no separable-compilation switch — each absence is a project invariant, not an oversight. Links are PRIVATE always (a device library's usage requirements are device-code include paths and `-x hip`; nothing linking it should inherit either), include directories arrive by linking the module that owns the kernel's bridge header (in gpumod that is `wwr.extension.parallel_for`), and separable compilation is OFF on every target in this tree. The macro's own header comment has the reasoning. Add a parameter when a real call site needs one.

It also creates no `::` alias, unlike the two macros below — its call site had none before the macro existed.

### Example

```cmake
calaman_add_gpu_device_library(
  NAME calaman.getrf.device
  SOURCES getrf_panel.cu
  LINK_PRIVATE wwr.extension.parallel_for)
```

> Illustrative: there is no such target yet. `deps/gpumod/src/extension/random_normal/CMakeLists.txt`
> is the real call site to copy from, and note that a device library added here
> must also be listed in `CROSS_CHECK_DEVICE_TARGETS` (`devtools/config.sh`) for
> `cross-backend-check.sh --device-only` to reach it.

---

## `CALAMAN_ADD_INTERFACE_LIBRARY`

Creates an INTERFACE library that exposes header files (`.h`/`.cuh`) to consumers. Automatically adds the standard project include directories and creates a `::` alias.

### Usage

```cmake
CALAMAN_ADD_INTERFACE_LIBRARY(
  NAME library_name
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `NAME` | required | Library target name |

### Always applied

- `add_library(${NAME} INTERFACE)`
- Include dirs: `INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>` + `$<INSTALL_INTERFACE:include>`
- `::` alias via `_calaman_create_alias`

### Example

```cmake
CALAMAN_ADD_INTERFACE_LIBRARY(
  NAME calaman.core.parallel_for)
```

---

## `CALAMAN_ADD_CXX_MODULE_LIBRARY`

Consolidates the common boilerplate for creating a C++23 module library.

### Usage

```cmake
CALAMAN_ADD_CXX_MODULE_LIBRARY(
  NAME my_library
  PRIMARY_INTERFACE my_module.cppm
  [PARTITIONS partition1.cppm partition2.cppm ...]
  [IMPLEMENTATION impl1.cpp impl2.cpp ...]
  [LINK_PUBLIC lib1 lib2 ...]
  [LINK_PRIVATE lib3 lib4 ...]
  [IMPORT_STD]
  [NO_CUDA_DEVICE_LINKING]
  [INCLUDE_CUDA_TOOLKIT PUBLIC|PRIVATE]
  [INCLUDE_DIRS_PUBLIC dir1 ...]
  [INCLUDE_DIRS_PRIVATE dir2 ...]
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `NAME` | required | Library target name |
| `PRIMARY_INTERFACE` | required | Primary module interface file (`export module M;`) |
| `PARTITIONS` | optional | Partition interface files (`export module M:P;`) |
| `IMPLEMENTATION` | optional | Module implementation files (`module M;`) |
| `IMPORT_STD` | flag | Enable `import std;` (sets `CXX_MODULE_STD 1`) |
| `NO_CUDA_DEVICE_LINKING` | flag | Sets `CUDA_SEPARABLE_COMPILATION OFF` and `CUDA_RESOLVE_DEVICE_SYMBOLS OFF` |
| `INCLUDE_CUDA_TOOLKIT` | `PUBLIC\|PRIVATE` | Add `CUDAToolkit_INCLUDE_DIRS` as SYSTEM includes |
| `INCLUDE_DIRS_PUBLIC` | optional | Directories to add as PUBLIC include paths |
| `INCLUDE_DIRS_PRIVATE` | optional | Directories to add as PRIVATE include paths |

### Example

```cmake
CALAMAN_ADD_CXX_MODULE_LIBRARY(
  NAME my_module.math.set_val
  PRIMARY_INTERFACE set_val.cppm
  IMPLEMENTATION set_val.cpp
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT PUBLIC
  LINK_PUBLIC cuda_runtime_api my_module::common
)
```

---

## Install and the CMake package

`calaman_install.cmake` emits every install rule and generates the package that
`find_package(calaman)` would find. The top-level `CMakeLists.txt` calls
`calaman_install_package()` once, last, when `CALAMAN_INSTALL` is on.

> **The tier is dormant, and one thing in this file is deliberately unfinished.**
> The layout assumptions inherited from gpumod are gone — the target sweep and the
> header install now treat `src/` as the single neutral tree it is, and the
> `cusolverMg` special case in `calamanConfig.cmake.in` went with them. What is
> *not* written is the line that makes a consumer able to resolve gpumod:
> `CALAMAN_INSTALL` defaults **OFF** because a calaman target's interface names
> gpumod targets that no installed package exports, and the fix depends on which
> of the two arrangements in `docs/architecture.md` section 2 is chosen. Expect
> the first real `CALAMAN_INSTALL=ON` to fail at generate time naming a `wwr.*`
> target; that failure is the accurate report, not a regression.

**Nothing has to be registered.** The function reads the buildsystem back and
installs every library target defined under `src/`, so a module added there is
in the package automatically and cannot be forgotten. The three rules below all
follow from one fact — a consumer of a C++23 module package *compiles the module
interface sources it installs*, because a BMI is not portable and so none can be
shipped:

1. Every compile requirement of a `.cppm` must reach the consumer. A `PRIVATE`
   include directory or define is not exported, which makes it a broken install
   rather than a private detail — grant it under `$<INSTALL_INTERFACE:>`, which
   leaves the in-tree build unchanged.
2. Module sources need per-target install destinations, because several modules
   are each rooted at a file with the same name (`interface.cppm`) and one shared
   destination would collide.
3. A header a module unit `#include`s is installed next to the installed sources,
   because that is where the relative include looks.

**`devtools/install-check.sh` is what verifies all of this**, by installing to a
throwaway prefix and building `example/consumer` against it. `cpp-tier.sh` cannot:
it never installs, so an export regression passes it cleanly. Every defect in the
list above was found that way in gpumod rather than reasoned about — which is the
argument for standing the tier back up here early rather than late.

### Forcing a device archive into an exported target

A module whose kernel lives in a separate `.device` library must force that
archive's members in, or the linker drops the
explicitly-instantiated device objects and consumers fail to link. Two things
make the force survive `install(EXPORT)`:

- Use the `$<LINK_LIBRARY:WHOLE_ARCHIVE,tgt>` genex, **not** the hand-written
  `-Wl,--whole-archive $<TARGET_FILE:tgt> -Wl,--no-whole-archive`: the
  hand-written form is copied into the export file unevaluated, and
  `$<TARGET_FILE:>` of a target the consumer does not have resolves to nothing.
- Spell the target **both** ways — `$<BUILD_INTERFACE:…tgt>` and
  `$<INSTALL_INTERFACE:…calaman::tgt>`. CMake does not namespace the name inside
  `LINK_LIBRARY` when it writes the export (it does for ordinary link entries), so
  the bare name matches nothing in the consumer and degrades to a plain `-ltgt`
  the linker cannot find.

Link it PUBLIC so it is also the ordinary link dependency (which is why the
`.device` target is not in the module's `LINK_PUBLIC`). `install-check.sh` is
what catches a regression here.

---

## Tests

Three macros, and which you want depends on whether the test *runs*.

`calaman_add_test_executable` builds a plain ctest-registered executable — used
by the compile-time tiers (`test/cuda`, `test/hip`, `test/gpu`), where the proof
is that the translation unit compiled and linked at all.

`calaman_add_gtest_executable` builds a GoogleTest binary, consolidating the
module properties, the shared `main.cpp` link list and libstdc++.

`calaman_add_gtest_suite_tests` registers that binary with ctest as **one entry
per suite**, plus a `SuiteListIsComplete` guard test that fails when a suite
exists in the binary but is missing from the CMake list. The list is
hand-maintained; the guard is what keeps it honest, so when it fails, add the
suite rather than deleting the guard. It runs
`calaman_check_gtest_suites.cmake`, whose path is resolved from
`${CMAKE_SOURCE_DIR}/cmake/`.

**Why per suite, not per binary or per case.** Measured on the math suite (276
cases, RTX 3080; date not recorded): one entry per binary is ~0.9s but
`ctest -R` then selects nothing useful and a whole binary reports as one
pass/fail; one entry per case is ~92s, almost all of it ctest's ~0.28s
per-entry process spawn (the binary itself starts in 0.09s); one entry per suite
is ~10s and keeps what the granularity is for — a `-R` that selects, and skips
that do not read as passes — at a tenth of the per-case wall clock.

Typed suites (`TYPED_TEST_SUITE`) register `Suite/0`, `Suite/1`, … one per type
in the `::testing::Types` list, in order; `TYPES` names them so an entry reads
`FooTests<double>` rather than `FooTests/1`, and must list the same types in the
same order as the alias the suites are instantiated over. A target whose typed
suites span more than one `Types` list needs one call per list (the cuBLAS tests
have eight suites over `IndexTypes` and five over `ElemsAndIndices`); the guard
still sees the union, because suite names accumulate on a global property and the
guard is deferred to the end of the directory scope.

Nothing invokes a test binary by path any more — `docker/compose.yaml` drives
`ctest`, so renaming a target does not need a change there.

### `REQUIRES_GPU`, and the `gpu` label

`calaman_add_gtest_suite_tests(... REQUIRES_GPU)` puts the `gpu` ctest label on
every entry it registers. Pass it when the binary needs a live device — it
allocates device memory, launches a kernel, or creates a vendor-library handle.
Seven targets do (`test/extension/{memory_buffer,runtime,rand,blas,solver,fft,
sparse}`); `test/gpu/conversions` does not, because host fp16/bf16 conversion
computes on the CPU.

The label is what lets `.github/workflows/ci.yml` build and test on a
GitHub-hosted runner, which has no card:

```
ctest --preset default    42 entries — everything
ctest --preset ci-cuda    12 entries — `-LE gpu`
ctest --preset ci-hip     12 entries — same, the other backend
```

**A label, not a `GTEST_SKIP`.** Nothing in `test/` gates on a device count,
and nothing should: an excluded test is named in the ctest output, where a
skipped one blends into a green run. It is the same mechanism as
`no_sanitizer`, which the `asan` preset and both compose sanitizer services
already exclude.

Two things to get right when adding one:

- **A second label must APPEND.** `set_tests_properties(... PROPERTIES LABELS
  x)` *replaces* the property, so on a `REQUIRES_GPU` target it silently strips
  `gpu`. Use `set_property(TEST ... APPEND PROPERTY LABELS x)` — see
  `test/extension/memory_buffer/CMakeLists.txt`, where two suites carry both
  `gpu` and `no_sanitizer`.
- **The `SuiteListIsComplete` guard is deliberately left unlabeled**, even for
  a `REQUIRES_GPU` target. `--gtest_list_tests` enumerates the registry without
  constructing a fixture, so it needs the binary to load but never touches a
  device — verified: all eight guards pass in a container started without
  `--gpus`. That keeps the hand-written suite lists honest on the runner too.
  If a target ever grows a static initializer that talks to the driver, label
  the guard rather than deleting it.

`cuda_compile_tests` is the one entry whose label depends on a cache variable,
for a reason unrelated to devices — it links the CUDA driver stubs and cannot
*load* without `libcuda.so.1`. It carries `gpu` by default, and with
`CALAMAN_CUDA_DRIVER_STUBS=ON` (the `ci-cuda` preset) it instead gets the
toolkit's stubs on `LD_LIBRARY_PATH` for that one test and runs. Its own
CMakeLists has the why, including why the stubs go in the build directory and
why `ENVIRONMENT_MODIFICATION` rather than `ENVIRONMENT`.
