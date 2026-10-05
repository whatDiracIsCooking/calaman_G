---
name: codestyle
description: >-
  This repo's C++ source conventions that no formatter or linter enforces —
  starting with `#include` style: quotes for a header this project owns, angle
  brackets for everything external (the standard library, vendor SDKs,
  LAPACKE/CBLAS, GoogleTest, and WarpWraps). Use when writing or reviewing
  `#include` lines under src/, experimental/ or test/, adding a new header or
  include root, or when the user asks about code style or include conventions.
  Not for comments and doc blocks — that's `docstyle`.
---

# Code conventions

Nothing in pre-commit or CI checks these — `ruff` covers Python only and there
is no clang-format hook — so they hold only if applied at the keystroke. Run
the check at the bottom after touching an `#include`.

## `#include`: quotes for ours, angle brackets for theirs

**The rule is ownership, not location on the include path.** A header whose
file lives in this repo's own trees (`src/`, `experimental/`, `test/`) is
`"quoted"`. Every other header is `<angled>`.

| Header comes from | Spelling | Examples |
|---|---|---|
| this repo | `"…"` | `"common/enums.h"`, `"error_handling/error_macros.h"`, `"lacpy_bridge.h"` |
| **WarpWraps** (`deps/WarpWraps/src/`) | `<…>` | `<runtime.h>`, `<complex.h>`, `<device_guard.h>`, `<extension/parallel_for/parallel_for.cuh>`, `<wrappers/math/math.cuh>`, `<wrappers/common/fp_types.h>` |
| the standard library | `<…>` | `<cstddef>`, `<vector>`, `<cfloat>` |
| vendor SDKs, LAPACKE, CBLAS, GoogleTest | `<…>` | `<lapacke.h>`, `<cblas.h>`, `<gtest/gtest.h>` |

**WarpWraps is external.** It is a git submodule under `deps/`, not part of
this project. It sits on the include path because `wwr.device` / `wwr::backend`
export its `src/` root with a plain `-I`, and an `-I` directory satisfies both
spellings. So a quoted WarpWraps include compiles, but it is still wrong. The
easy way to miss this: a header like `"runtime.h"` *looks* local. Find out
where the file lives before choosing quotes.

### Spelling a header this project owns

- **Same directory as the includer:** bare name — `"ladiv.h"` from
  `src/lapack/ladiv/ladiv.cu`, `"enums.h"` from inside `src/common/`.
- **Anywhere else:** root-relative, by the path its include root makes
  resolve — `"common/elem_ops.cuh"`, `"lapack/ladiv/ladiv.h"` (both off the
  `src/` root).
- **Never a `../` climb.** If a header is not reachable root-relative, the
  target is missing an include root; fix the CMake, not the spelling.
- A non-module header that a `.cppm` includes from its global module fragment
  needs its include root exported `PUBLIC`/`INTERFACE`, not `PRIVATE`, or the
  consumers that rebuild the BMI cannot find it.

### Why angle brackets for WarpWraps is safe — `<complex.h>`

WarpWraps ships a `complex.h`, and so do the C standard library and libc++.
`<complex.h>` still resolves to WarpWraps' copy, because its root is a plain
`-I` (not `SYSTEM`/`-isystem`), and clang searches `-I` directories before
the toolchain's own system directories. Two things would break that:

- `add_subdirectory(WarpWraps SYSTEM)` in `deps/CMakeLists.txt`, or
- WarpWraps marking its include root `SYSTEM`.

Either one turns WarpWraps' root into an `-isystem` directory, and the lookup
order between it and libc++'s wrappers then depends on the toolchain. If you
make either change, rebuild both backends and check that `wwrFloatComplex`
still resolves.

## Check

This prints every quoted include whose target does not exist in this repo,
which means it is external and should be angled. Clean output means the tree
complies.

```bash
grep -rhoE '#include "[^"]+"' src experimental test | sort -u |
  sed -E 's/#include "(.*)"/\1/' | while read -r h; do
    find src experimental test -path "*/$h" -print -quit | grep -q . ||
      echo "external but quoted: $h"
  done
```

The other direction, an angled include of one of our own headers, shows up as
a name that `find src experimental test -path "*/<name>"` locates.
