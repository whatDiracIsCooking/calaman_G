# Architecture notes

Vendor facts and language constraints this project has to live with — things
nobody here *decided*, but which a reader will otherwise rediscover the hard
way: a header that poisons a macro, a wavefront-width divergence, a linkage rule
that decides where a declaration may live — together with the design decisions
this project made in response, which belong on the same record.

Headers under `src/` state rules; this file says why, and on what it was
measured. That separation exists because a header claims to describe code as it
is *now* — a sentence like "verified on ROCm 7.2.4" sitting in one cannot be
told apart from a sentence re-checked yesterday. Here, every claim carries a
date and a toolchain, so its staleness is visible.

**Toolchain for every claim below unless a section says otherwise:** CUDA 13.0
(nvcc), clang 20.1.8 with libc++, ROCm 7.2.4, AMD target `gfx1200`, NVIDIA
target `sm_86`. Dated 2026-09-23.

A section is not re-verified when the toolchain moves. If you upgrade and a
claim here matters to you, re-check it and update the date.

Sections are cited from code by number, so **append rather than insert**.

---

## 1. `gpurandState` is not `gpurandStateXORWOW` on HIP

**The divergence.** cuRAND makes the default state an alias
(`typedef struct curandStateXORWOW curandState;` — one type, two names).
hipRAND emits a **separate struct per generator** over a shared
`rocrand_state_xorwow` base, so `hiprandState` and `hiprandStateXORWOW` are two
distinct C++ types with the same layout.

**Consequence.** A `gpurandState*` where a `gpurandStateXORWOW*` is wanted
compiles on CUDA and fails on HIP. Pick one spelling and keep it. Pinned by
`test/hip/hiprand_kernel.cppm` and `test/gpu/rand.cppm`, which assert the two
backends' opposite answers separately.

**Why `rand_state_bridge.h` (`src/extension/bridge/`) declares
`::curandStateXORWOW`, not `::curandState`.**
`curandState` is itself a typedef, and a typedef cannot be forward-declared —
only the struct can. On CUDA the two are one type, so the alias names the same
type `rand.cuh` and `rand.cppm` do. The asymmetry is deliberate: on HIP the two
are genuinely distinct, so the HIP branch declares `hiprandState` and nothing
else.

Forward-declaring is the vendor's own idiom on CUDA — `curand_kernel.h`
declares `struct curandStateXORWOW;` itself before defining it. Both branches
were checked to coexist with the same name arriving through
`import wwr.rand` in one TU: clang merges the two alias declarations
because both resolve to the same entity, the vendor struct, which lives in the
global module either way. If a vendor renamed the underlying struct the two
would name different types and clang would reject it as a typedef redefinition
— so this fails loudly rather than silently.

## 2. Cooperative groups: one namespace, divergent types

`src/cooperative_groups.cuh` wraps nothing: it resolves the include and defines
no names of its own — both vendors put cooperative groups in
`namespace cooperative_groups` and agree on the spellings inside, so a
forwarding layer would only rename each name to itself. The header is the
`#include` switch and nothing more.

The divergences are the caller's to handle:
`ballot()` is 32-bit on CUDA and 64-bit on HIP, so storing one in an `unsigned`
is silently wrong on a wave64 part; `thread_rank()` and `num_threads()` vary by
group type on CUDA but not on HIP; tiles are bounded by `WWR_WARP_SIZE`;
`grid_group` has five portable members, CUDA's `block_rank()` family having no
HIP counterpart; and `thread_block::group_dim()` is static on CUDA but a
non-const member on HIP.

## 3. Complex construction goes through `make_gpu*Complex`

`cuFloatComplex` is `float2`, a plain aggregate; `hipFloatComplex` is a
`HIP_vector_type<float, 2>` class. Brace-initialising `gpuFloatComplex{re, im}`
is therefore not the same operation on both backends, and the vendor
make-functions are the portable way to build one.

The float-to-half conversions need no such treatment: `__float2half` and
`__float2bfloat16` are spelled identically by both vendors. They are re-exposed
under `gpu*` names anyway, for the layering reason in §5.

## 4. Forwarding templates where `WWR_FUNCTION` cannot reach

`WWR_FUNCTION` binds a reference straight to the backend's function
(`inline constexpr auto& gpuX = ::cuX;`), so the signature is never restated and
cannot drift. A function reference carries no default arguments and **cannot
name an overload set**, which rules it out in three places:

- **`rand.cuh`'s generators.** `curand_normal` and friends are an *overload set*
  on CUDA (one per state type) and a *function template* on hipRAND (one
  template, constrained by a `check_state_type` static_assert). A reference can
  name neither, so each is a thin `__device__` template over the state type.
- **`gpuMalloc`.** `hipMalloc` has a `template<class T>` overload, so the
  reference is given an explicit type to select one:
  `inline constexpr gpuError_t (&gpuMalloc)(void**, std::size_t) = ...;`
- **Signature divergences** (§8) — a hand-written forwarding function.

## 5. const-correctness divergences between the vendors

Resolved inside the gpu* layer, never above it. In each case the neutral `gpu*`
signature is the const-correct one, and the other backend gets a forwarding
function with a `const_cast` into an API that only reads those arguments.

| Neutral name | Divergence |
|---|---|
| `gpublas*getrsBatched`, `gpublas*getriBatched` | hipBLAS declares the input arrays (and `getriBatched`'s pivots) non-const where cuBLAS declares them const. The `gpublas*` signature keeps cuBLAS's; HIP forwards with a `const_cast`. |
| `gpurandGetScrambleConstants32`/`64` | The neutral signature takes hipRAND's const-correct `const unsigned int**` / `const unsigned long long**`; on CUDA these are forwarding functions around cuRAND's non-const signature. |

Also name-level, not signature-level: hipBLAS has a single status-to-string
function, so `gpublasGetStatusName` and `gpublasGetStatusString` both map to
`hipblasStatusToString`. Neither cuSOLVER nor hipSOLVER has one at all, so
`gpusolverGetStatusName`/`String` are hand-written switches, one per backend,
over each backend's own differently-sized enumerator set.

## 6. Enumerator values differ even where names agree

`GPURAND_RNG_PSEUDO_DEFAULT` is **100** on cuRAND and **400** on hipRAND; every
RNG type is offset the same way. Status, ordering and direction-vector-set
values happen to agree. **Use the names; never store or compare the numbers.**

## 7. hipRAND's bare `printf`, and include order

`hiprand_kernel.h` → `hiprand_kernel_rocm.h` → `rocrand/rocrand_kernel.h` →
`rocrand/rocrand_mtgp32.h` calls bare `printf` without declaring it. A real
`-x hip` compile usually drags a declaration in through `hip_runtime.h` first,
so the failure — `error: use of undeclared identifier 'printf'` — appears only
when the hipRAND header is reached first.

`<cstdio>` is therefore included ahead of it in both `src/rand.cuh` and
`src/hip/hiprand_kernel.cppm`, so the consuming TU's include order cannot
matter.

## 8. `import std;` is not available to a device TU

A `.cu` (or `-x hip` device-compiled) translation unit imports no modules at
all, by this project's convention — neither `src/extension/init_state/init_state.cu`
nor `src/extension/random_normal/random_normal.cu` `import`s anything. That is what
forces the `.cuh` half of the gpu* layer to exist at all: the same `gpu*` names have
to be reachable by `#include`, with the backend picked from the compiler's own
device-compile macro rather than from a CMake define.

The types are the **same types** the modules export under the same names, so a
buffer allocated by host code that imports `wwr.rand` is exactly what a
kernel naming `gpurandState` expects, and an `extern template` declared in a
`.cppm` links against a definition compiled in a `.cu`.

## 9. `<array>` before any HIP header

`amd_detail/amd_hip_vector_types.h` (pulled in transitively by most HIP
headers) does `#include "hip/amd_detail/host_defines.h"` immediately before
`#include <array>`. Outside HIP device-compilation mode, `host_defines.h`'s
non-HCC branch defines `__noinline__` as an **empty** object-like macro. If
`<array>` — and libc++'s `__config` through it — is first included after that
point in the TU, `__has_attribute(__noinline__)` inside `__config` expands to
`__has_attribute()`, zero arguments, which clang rejects.

Including `<array>` ourselves *before* the HIP header sidesteps it: `__config`
is fully processed with the real, unpolluted `__noinline__`, and the later
re-inclusion from `amd_hip_vector_types.h` is a no-op via the include guard.

**This is load-bearing.** A bare `#include <hip/hip_complex.h>` fails this way;
pre-including `<array>` is necessary and sufficient. The pre-include is not a
tidy-up candidate — every `#include <array>` in a `src/hip` global module
fragment is there for this reason and must stay first.

Affected: `hip_bf16`, `hip_complex`, `hip_fp4`, `hip_fp6`, `hip_fp8`,
`hip_fp16`, `hipblas`, `hipblaslt`, `hipfft`, `hipfftXt`, `hipsolver`,
`hipsparse`.

## 10. `<algorithm>` as well, for hip_fp8

`amd_hip_fp8.h` includes `amd_hip_bf16.h` before `amd_hip_fp16.h`.
`amd_hip_bf16.h` brings in `amd_detail/device_library_decls.h`, which
`#define`s `__local` as `__attribute__((address_space(3)))`. `amd_hip_fp16.h`
then does the TU's first `#include <algorithm>` — and with `__local` already
poisoned, libc++'s `<algorithm>` internals (`_Traits::__local(...)` in
`__algorithm/copy.h` and friends) fail to parse.

Pre-including `<algorithm>` gets it fully processed, via its include guard,
before `device_library_decls.h` ever runs. `<array>` alone is **not** sufficient
for this module, unlike `hip_complex` / `hip_fp16` / `hip_bf16`.

## 11. hip_fp4 and hip_fp6 cannot share a translation unit

Both `amd_detail/amd_hip_fp4.h` and `amd_detail/amd_hip_fp6.h` define
`internal::half_to_f16`, `internal::half2_to_f16x2`,
`internal::hipbf16_to_bf16` and `internal::hipbf162_to_bf16x2` as **non-inline
`static`** functions in the same `internal` namespace. Combining both headers
in one TU is a redefinition error.

So `wwr.hip.hip_fp4` and `wwr.hip.hip_fp6` must never appear in the same
translation unit's global module fragment. `hip_fp4.cppm` includes only
`hip_fp4.h`; `hip_fp6.cppm` includes only `hip_fp6.h`.

## 12. `static inline` vendor functions need forwarding wrappers

A function declared `static inline` in the global namespace has
internal-to-the-TU linkage and cannot be re-exported by a `using` declaration.
Both vendors do this for whole families, so the wrapping modules provide thin
forwarding functions with the same names instead:

- `cuComplex.h` / `amd_hip_complex.h` — every function.
- The `__nv_cvt_*` / `__hip_cvt_*` fp4/fp6/fp8 conversion families.
- HIP's `hipMallocAsync`/`hipMallocFromPoolAsync` typed overloads and the
  `hipBindTexture*`/`hipUnbindTexture` templates.

Clang's diagnostic when this is missed is
`using declaration referring to 'X' with internal linkage cannot be exported` —
even when an `extern "C"` overload with external linkage also exists.

`hip_runtime_api.cppm` has a second, distinct reason for the same treatment:
for `hipMemcpyToSymbol`, `hipGetSymbolAddress`, `hipLaunchCooperativeKernel` and
the `hipOccupancy*` family, the convenience overload is a non-static function
*template*, so `using ::name;` compiles — but `test/hip/hip_runtime_api.cppm`'s
`WWR_LINK_CHECK(name)`, which takes `&name` with no arguments or target type, cannot
disambiguate it from the non-template overload. Those ~13 names get forwarding
functions too.

`__HIP_DISABLE_CPP_FUNCTIONS__`, defined before the include in
`hip_runtime_api.cppm`, is the header's own escape hatch covering 5 of these
(`hipMalloc<T>`, `hipMallocPitch<T>`, `hipHostMalloc<T>`, `hipHostAlloc<T>`,
`hipMallocManaged<T>`); the rest are declared outside its guard.

## 13. A `const` member of class type breaks `device_functor`

`parallel_for`'s `device_functor` concept requires the functor to be trivially
copyable **and** not copy-assignable. The second half is what enforces
immutability: under CUDA the kernel receives the functor via
`WWR_GRID_CONSTANT` (constant memory shared across the grid), and writes to
constant memory are undefined behaviour on either backend.

Marking **every** member `const` is the obvious way to satisfy
"not copy-assignable", and it is wrong. **A `const` member of class type makes
the enclosing class non-trivially-copyable under clang**, which then fails the
concept's other requirement. A `const` member of scalar type, pointers
included, is unaffected.

Checked on clang 20.1.8, both libc++ and libstdc++, C++20 and C++23, with no
`-x hip` involved: it is a plain C++ property, not a backend quirk.

**The rule.** Keep at least one `const` **scalar** member — that is what
deletes the copy assignment — and leave class-type members non-`const`.
Nothing is lost, because `parallel_for_kernel` takes the functor as
`WWR_GRID_CONSTANT const Functor` and `operator()` is const-qualified, so the
kernel cannot write it by either route.

`src/wrappers/fill/fill.cu`'s two functors are the worked example. It cost a
silently HIP-broken module to find, because nvcc accepts what clang rejects
and the CUDA build stayed green throughout.

## 14. A module's purview gives a declaration module linkage

A namespace-scope name declared in a module's **purview** and not exported has
module linkage: clang mangles it with the module name attached, so it can never
resolve to a definition compiled in a plain (non-module) translation unit. The
same declaration two ways:

```
purview:  U probe::detail::impl_fn@probe.purview(int)  -> undefined reference
GMF:      U probe::detail::impl_fn(int)                -> links
```

The device-compiled TUs are plain TUs — nvcc and `-x hip` cannot compile a
module unit — so a declaration they must bind to has to have ordinary external
linkage, and the **global module fragment** is what gives it that. This is
specific to module units: a plain `.cpp` calling the same function needs no
such care, since a bare forward declaration in it already has external linkage.

That constraint is also why a bridge header's types come from `#include`-only
headers rather than an `import`: a GMF can `#include` but cannot `import`. The
the gpu* layer's bridge headers give the *same* types the modules export, so exported
wrappers pass arguments straight through with no conversion and the device side
needs no cast.

## 15. FP atomics on AMD, and `-munsafe-fp-atomics`

The common atomics (`atomicAdd`, `atomicCAS`, `atomicExch`, `atomicMin`/`Max`,
`atomicAnd`/`Or`/`Xor`, `atomicSub`, `atomicInc`/`Dec`) are spelled identically
in the global namespace on CUDA and HIP, with the same signatures, and are
declared by the vendor runtime header `runtime.cuh` already switches. So
gpumod wraps none of them, for the reason §2 gives: a forwarding
function per name would only rename each name to itself. A device TU that
includes `runtime.cuh` calls them bare; `test/gpu/atomics.cu` is what
pins that the common widths resolve under both front ends, nvcc being the
permissive one.

The one divergence is not in the source at all but in AMD's *code generation*
for floating-point atomics, and it is a compile flag. By default clang lowers a
floating-point atomic AMD has no safe native instruction for to a CAS loop, so
`atomicAdd` on `float`/`double` is correct on both backends with no extra flags
— what `test/gpu/atomics.cu` builds. `-munsafe-fp-atomics` is an opt-in HIP
device-compile flag that makes AMD emit native FP-atomic instructions instead:
faster, but with weaker guarantees — it can flush denormals, and on fine-grained
memory it can silently drop the update rather than fault. gpumod never sets it;
a TU that wants the trade-off passes it on its own device library (the
`wwr_add_gpu_device_library` target that owns the kernel), never on a target
that reaches non-device code.

One-sided atomics stay exposed too, for the same reason: the scoped CUDA forms
(`atomicAdd_block` / `_system`), CUDA's `half`/`half2` `atomicAdd`, and AMD's
`unsafeAtomicAdd` / `atomicAddNoRet` have no portable counterpart — a TU that
knowingly wants one names the vendor function directly.

## 16. WMMA: one surface, two namespaces and a wave-dependent fragment

Measured 2026-09-25 against rocWMMA 2.2.0 (ROCm 7.2.4) and CUDA 13.0.88 at
`sm_80` rather than the file-wide `sm_86`; the CUDA counts below are the same
for every `sm_70`+ target.

AMD wrote rocWMMA to be source-compatible with NVIDIA's WMMA, so `fragment`,
the `matrix_a`/`matrix_b`/`accumulator` and `row_major`/`col_major` tags,
`layout_t`, `fill_fragment`, `load_matrix_sync`, `store_matrix_sync` and
`mma_sync` are spelled identically. The namespaces are not: `nvcuda::wmma`
against `rocwmma`. That is one name more than §2's case, which is why
`src/wmma.cuh` defines the `gpuwmma` alias where `cooperative_groups.cuh`
defines nothing. Two of the divergences below are silent, and both bite code
that never reads either file.

**`gpuBfloat16` is not a WMMA element type on HIP.** `bf16.cuh` aliases it to
`__nv_bfloat16` (CUDA) and `__hip_bfloat16` (HIP), but rocWMMA's `bfloat16_t`
is the *older* `hip_bfloat16`, a distinct type. Both backends do 16x16x16 bf16;
only CUDA does it with the type this layer hands out. The HIP failure is a
template error about an undefined `rocwmma::PackTraits<__hip_bfloat16>`, which
names neither bfloat16 nor WMMA. `gpuHalf` has no such problem — it is `__half`
on both, and rocWMMA's `hfloat16_t` is `__half` too.

**`fragment::num_elements` changes between HIP's two compile passes.** It is the
per-lane share of the tile, so it follows the warp width: 16 (`matrix_a`) and 8
(`accumulator`) on CUDA, and on HIP whatever `__AMDGCN_WAVEFRONT_SIZE__` says in
the pass being compiled — 4 and 4 in the wave64 host pass, 8 and 8 in the
gfx1200 device pass of the same TU. This is the same warp-size trap surfacing
in a type's member: a `static_assert` on a literal count cannot hold in both passes,
and host code sizing a buffer from it is wrong with nothing to diagnose it.
Multiplying by a warp width does not rescue it either: `WWR_WARP_SIZE` is the
configure-time 32 in both passes, and clang 20 deprecates
`__AMDGCN_WAVEFRONT_SIZE__` outright — *"compile-time-constant access to the
wavefront size will be removed in a future release"*. So `test/gpu/wmma.cu` pins
what holds in both passes without naming a width: on HIP the two fragments hold
the same per-lane share, and neither is CUDA's 16.

## 17. HIP device libraries: how a `.cu` is compiled

Measured 2026-09-25 against clang 20.1.8 and ROCm 7.2.4 at
`--offload-arch=gfx1200`.

Under the HIP backend the project enables no CUDA language — the top-level
`CMakeLists.txt` decides that before `project()`, and a ROCm-only box has no CUDA
toolkit. So the steps that turn a `.cu` into device code are arranged by hand, in
`cmake/wwr_add_gpu_device_library.cmake`.

**The `.cu` extension is overridden back to CXX.** CMake maps `.cu` to the CUDA
language by extension; with CUDA disabled that cannot stand, so each source gets
`LANGUAGE CXX`. An explicit `LANGUAGE` beats extension detection, and the tree
then configures and builds with CUDA absent — checked with a `project(CXX)`-only
probe compiling a `.cu` as CXX.

**`-x hip` wins by being last.** `hip::device` injects `-x hip
--offload-arch=gfx1200` through `INTERFACE_COMPILE_OPTIONS`, gated on
`$<COMPILE_LANGUAGE:CXX>` — which is why the sources must be CXX (above) for it
to reach them. The compile line ends up `-x c++ ... -x hip
--offload-arch=gfx1200`, and clang honours the **last** `-x`, so the leading
`-x c++` the `LANGUAGE` property adds is inert. Seen in `build-hip`'s
`compile_commands.json`.

**The device link step needs ROCm's `lld`.** A real `-x hip` compile ends in an
`amdgcn-link` step that invokes its device linker as the bare name `lld`, which
the system clang does not ship — only ROCm's bundled LLVM does. The macro points
the driver's tool search at it with `-B<prefix>/llvm/bin`, deriving `<prefix>`
from `hip::amdhip64`'s include directory rather than hardcoding a ROCm version.

**The data-layout linker warning is benign.** `amdgcn-link` emits *"Linking two
modules of different data layouts"* once per ROCm device-bitcode module (`hip.bc`,
`ocml.bc`, `ockl.bc`, the `oclc_*.bc`) — ~10 lines per `.cu`. It is a
compiler-version skew, not a defect: the system clang-20 writes address space 8
as `p8:128:128`, while ROCm's `/opt/rocm/amdgcn/bitcode` is built by its own
newer LLVM (clang-22), which writes `p8:128:128:128:48` — the buffer-resource
index width LLVM 21+ added. The layouts are otherwise identical and the linked
code is correct. There is no finer diagnostic than the `-Wlinker-warnings` group,
so the macro sets `-Wno-linker-warnings`, scoped to the HIP branch and to device
`.cu` libraries alone. It lapses once the two LLVMs converge — drop the flag then
and confirm the warning is gone.

**Separable compilation is OFF** on these libraries, as everywhere in the tree:
`parallel_for.cuh`'s kernel is self-contained, so no device symbol crosses a TU
boundary and relocatable device code would cost link time to buy nothing.
