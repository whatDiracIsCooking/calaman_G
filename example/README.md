# Examples

**Empty on purpose: there is nothing to demonstrate yet.** `src/` holds no
modules, so an example here could only show a build system consuming nothing, and
a program that compiles while asserting nothing is worse than no program — it is
a tier that reports green.

Two kinds of example belong here, and they are wired differently.

## In-tree examples

A small program per idea, written the way a user would write it — `import
calaman.<x>;`, the real API, no test-only helpers. Each gets a directory with its
own `CMakeLists.txt`, plus an `example/CMakeLists.txt` that adds them, and the
top-level `add_subdirectory(example)` line (commented out today) is uncommented in
the same commit.

Add them **unconditionally**, not behind an option: `devtools/cross-backend-check.sh`
compiles the compile-time tier for the *other* backend, and an example excluded
from that configuration is an example that stops compiling for AMD without anyone
noticing.

`deps/gpumod/example/` is the model, and it is in this checkout — its
`warp_reduce` example is the pattern for one that needs a `.cu` device library.

## `example/consumer/` — the package tier's other half

A **standalone** project that knows nothing about this source tree and reaches
this library through `find_package(calaman)` alone.
`devtools/install-check.sh` installs into a throwaway prefix and then configures
and builds that project against it, which is the only thing that proves the
installed package is usable: a `.cppm` with a `PRIVATE` include directory builds
in-tree, installs cleanly, and fails in every consumer.

It does not exist yet, and the script says so rather than pretending. Two things
have to happen first:

1. `src/` has to install something — `CALAMAN_INSTALL` is OFF until the
   two-package export question is settled (`docs/architecture.md` §2).
2. `example/consumer/` must be a *separate* project (its own
   `cmake_minimum_required` and `project()`), reached only through
   `CMAKE_PREFIX_PATH`. It is deliberately **not** added by
   `example/CMakeLists.txt` — building it as part of this tree would prove
   nothing, because in-tree targets resolve without the install.

`deps/gpumod/example/consumer/` is the working version of exactly this, and the
one to copy when the time comes.
