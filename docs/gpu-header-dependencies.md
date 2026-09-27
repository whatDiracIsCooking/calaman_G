# `src/` header dependency tree

The `.h` and `.cuh` switch-point headers of the `gpu*` layer (directly under
`src/`), plus the two bridges (now under `src/extension/bridge/`), as opposed to
the `.cppm` modules. This records their include graph as it stands, so the
flatness is visible at a glance and a new edge stands out in review. The bridges
live in the extension layer but are documented here because they are switch
points too — they reach `selected_backend.h` directly, and count among the four.

See also `src/README.md` ("The switch points"), which explains *why* the graph
has this shape; this file only states *what* it is.

## The graph

`selected_backend.h` is the leaf everything rests on. The device `.cuh` files
reach it through `device_guard.h`, which adds the "must be a device pass"
`#error` guard a bridge must not have. Four reach `device_guard.h` directly;
`cooperative_groups.cuh` and `wmma.cuh` reach it through `runtime.cuh`,
whose `WWR_WARP_SIZE` they also want — a portable tile size for one, a wave
index for the other, both because the API is a whole-warp collective.
**No bridge includes another bridge, and every `.cuh`-to-`.cuh` edge stays
inside a single target** — both are into `runtime.cuh`, and all three
files are `wwr.device`, so neither leaks an include path across targets,
which is the concern that keeps every other `.cuh` rooted directly at
`device_guard.h`.

The two bridges — `gpu_stream_bridge.h` and `rand_state_bridge.h` — reach
`selected_backend.h` *directly*, not through `device_guard.h`: a bridge compiles
in a host TU and so must not carry the "must be a device pass" `#error`. They
appear below as leaves of `selected_backend.h` alongside `device_guard.h`. Both
live in `src/extension/bridge/` and are carried by the `wwr.extension.bridge`
INTERFACE target; their consumers are `extension/parallel_for/parallel_for.cuh`,
`extension/init_state/init_state_bridge.h` and
`extension/random_normal/random_normal_bridge.h`.

```
selected_backend.h        no #includes — the leaf the switch/bridge layer rests on
├── device_guard.h         + the "is this a device pass?" #error guard; no vendor header
│   ├── runtime.cuh  + <cuda_runtime.h>            | <hip/hip_runtime.h>
│   │   ├── cooperative_groups.cuh
│   │   │                  + <cooperative_groups.h>      | <hip/hip_cooperative_groups.h>
│   │   └── wmma.cuh       + <mma.h>                     | <rocwmma/rocwmma.hpp>
│   ├── complex.cuh        + <cuComplex.h>
│   │                      | <hip/hip_complex.h>
│   ├── fp16.cuh           + <cuda_fp16.h>               | <hip/hip_fp16.h>
│   ├── bf16.cuh           + <cuda_bf16.h>               | <hip/hip_bf16.h>
│   └── rand.cuh           + <curand_kernel.h>           | <cstdio> <hiprand/hiprand_kernel.h>
├── gpu_stream_bridge.h   + <cuda_runtime_api.h>        | <hip/hip_runtime_api.h>
└── rand_state_bridge.h   forward-declares the vendor struct; no vendor header

gpu_backend.h             no #includes — independent; consumed only by the .cppm modules
```

The two columns after each `+` are the CUDA branch (`WWR_SELECTED_CUDA`) and
the HIP branch (`WWR_SELECTED_HIP` / the `#else`); a translation unit sees
exactly one. `complex.cuh`, `fp16.cuh`, `bf16.cuh`, `runtime.cuh`,
`cooperative_groups.cuh`, `wmma.cuh`, `rand.cuh` and `gpu_stream_bridge.h` pull
vendor headers; `rand_state_bridge.h` pulls none (it forward-declares the vendor
struct), and `device_guard.h` pulls none — it carries only the guard. That guard
is a check on `__CUDACC__` / `__HIP__` / `__HIPCC__`, separate from the backend
selection: it answers "is this a device pass?", not "which backend?", which is why
it lives in `device_guard.h` and not in `selected_backend.h` (a bridge includes
the latter from a host compile and must not `#error` there — which is exactly why
the two bridges reach `selected_backend.h` directly rather than through
`device_guard.h`).

## Consumers (reverse edges)

Who reaches each header from outside `src/`. `selected_backend.h` is reached
transitively — through `device_guard.h` internally, and through the two bridges
in `src/extension/bridge/`.

| Header | Included by | CMake target that carries it |
|---|---|---|
| `selected_backend.h` | (internal — `device_guard.h`; and the two bridges) | — (header-only, no target of its own) |
| `device_guard.h` | (internal only — `runtime.cuh`, `complex.cuh`, `fp16.cuh`, `bf16.cuh`, `rand.cuh`) | — (header-only, rides each `.cuh`'s target) |
| `gpu_backend.h` | `blas.cppm`, `bf16.cppm`, `complex.cppm`, `fp16.cppm`, `rand.cppm`, `runtime_api.cppm`, `solver.cppm` | each module's own target |
| `gpu_stream_bridge.h` | `extension/parallel_for/parallel_for.cuh`, `extension/init_state/init_state_bridge.h`, `extension/random_normal/random_normal_bridge.h` | `wwr.extension.bridge` |
| `rand_state_bridge.h` | `extension/init_state/init_state_bridge.h`, `extension/random_normal/random_normal_bridge.h` | `wwr.extension.bridge` |
| `runtime.cuh` | `extension/parallel_for/parallel_for.cuh` (and, internally, `cooperative_groups.cuh`) | `wwr.device` |
| `cooperative_groups.cuh` | (none yet — the warp-reduction example will be its first caller) | `wwr.device` |
| `wmma.cuh` | (none yet — only `test/gpu/wmma.cu` compiles it) | `wwr.device` |
| `complex.cuh` | `extension/random_normal/random_normal.cu` | `wwr.device` |
| `fp16.cuh` | `extension/random_normal/random_normal.cu` | `wwr.device` |
| `bf16.cuh` | `extension/random_normal/random_normal.cu` | `wwr.device` |
| `rand.cuh` | `extension/init_state/init_state.cu`, `extension/random_normal/random_normal.cu` | `wwr.rand.device` |

`runtime.cuh`, `cooperative_groups.cuh`, `wmma.cuh`, `complex.cuh`, `fp16.cuh` and
`bf16.cuh` share the `wwr.device` target; `rand.cuh` sits in a separate `wwr.rand.device`
target on purpose (an RNG device TU should not be forced to carry the
runtime/warp-size machinery). That target split is why `device_guard.h` — not
`runtime.cuh` — is where the shared guard lives: it carries only the guard
and `selected_backend.h`, no vendor headers and no target-specific content, so
each `.cuh` can include it without one target's include path or vendor headers
leaking into another's consumers. Routing a `.cuh` in one target through a `.cuh`
in another would do exactly that leaking, which is why every cross-target pair is
kept apart. The two `.cuh`-to-`.cuh` edges, `cooperative_groups.cuh` and `wmma.cuh` →
`runtime.cuh`, are within `wwr.device`, so they carry no such leak: a
consumer of any of the three already links that one target.
