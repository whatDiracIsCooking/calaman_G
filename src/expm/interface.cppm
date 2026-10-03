/**
 * @file interface.cppm
 * @brief Primary interface for calaman.expm -- the matrix exponential exp(A) by
 *        scaling and squaring with a diagonal Pade approximant (Higham's
 *        Algorithm 2.3)
 *
 * exp(A) for a square column-major A, by Higham's scaling-and-squaring method
 * (Higham, "The Scaling and Squaring Method for the Matrix Exponential
 * Revisited", SIAM J. Matrix Anal. Appl. 26(4), 2005). The driver lives in
 * :expm; see expm.cppm and the README for the algorithm, the degree-13 nested
 * form, the cost table and the (deliberately external) balancing recipe.
 *
 * Partitions:
 * - :plan         degree/scaling selection: pade_theta, ExpmPlan, expm_plan
 * - :norm1        the induced matrix 1-norm, matrix_norm1
 * - :buffer_size  device-workspace sizing: pade_bufferSize, expm_bufferSize
 * - :pade         the unscaled [m/m] Pade approximant r_m(A)
 * - :expm         the scaling-and-squaring driver
 * - :detail       shared internals (NOT re-exported): as_element, kMaxDim and
 *                 the two workspace layouts
 *
 * STRUCTURE. expm is a HOST COMPOSITION over wrapped BLAS (gemm/geam), the
 * wrapped LU solve (getrf/getrs) and four fused kernels of its own (expm.cu,
 * reached through expm_bridge.h). The matrix products are the OUTERMOST
 * WarpWraps layer that does the job -- the type-safe wrappers in
 * wwr.wrappers.blas / wwr.wrappers.solver (CLAUDE.md, "prefer the outermost
 * layer"). It owns a launcher bridge (expm_bridge.h), a device-compiled unit
 * (expm.cu) and an explicit-instantiation unit (instantiations.cpp), the same
 * module/.cu split calaman.lacpy / calaman.gebal use.
 *
 * TEMPLATED OVER ALL FOUR element types (float, double, and the two complex
 * types), constrained by wwr::usual_fp. The Pade coefficients are real even for
 * a complex matrix, so the only complex constants needed are built at run time
 * by :detail's as_element (wwr.complex host constructors, legal in a module
 * purview though not in a GMF -- calaman.common's constants.h documents why).
 *
 * NO gemm3m. The reference this was ported from offered cublasGemm3m (Gauss's
 * 25%-fewer-flops complex product) behind an option; it is a cuBLAS-only call
 * with no hipBLAS counterpart (it lives in wwr.cuda.cublas_v2, not the neutral
 * layer), so routing through it would break the backend-neutrality this project
 * is built on. Every matrix product goes through the portable wwr::gemm.
 *
 * STATUS. Returns calaman::Status (calaman.error_handling), via CLM_TRY -- the
 * cross-domain convention the rest of src/ returns. A failed BLAS or solver call
 * carries its OWN domain's code; the genuinely host-side checks stay BLAS-domain
 * outcome codes (a bad argument WWRBLAS_STATUS_INVALID_VALUE, an undersized
 * workspace WWRBLAS_STATUS_ALLOC_FAILED), both of which convert to Status
 * implicitly.
 *
 * Usage:
 *   import calaman.expm;
 *   import wwr.blas;     // wwrblasHandle_t, wwrblasCreate
 *   import wwr.solver;   // wwrsolverDnHandle_t, wwrsolverDnCreate
 *   import wwr.runtime_api;  // wwrStream_t
 *   // cublas, cusolver bound to the SAME stream; d_A, d_expA: n-by-n device
 *   std::size_t bytes = 0;
 *   calaman::expm_bufferSize<double>(cusolver, n, &bytes);
 *   // d_work: bytes of 256-aligned device scratch; d_info: device int[2]
 *   calaman::ExpmPlan plan{};
 *   calaman::expm<double>(cublas, cusolver, stream, n, d_A, n, d_expA, n,
 *                         d_work, bytes, d_info, &plan);
 */

export module calaman.expm;

// :detail is imported WITHOUT re-export: its names stay module-internal (shared
// only among the partitions), exactly as they were a non-exported namespace
// before the split. The plain import also makes it reachable from the primary,
// as every interface partition of the module must be.
import :detail;

export import :plan;
export import :norm1;
export import :buffer_size;
export import :pade;
export import :expm;

// expm()/pade()/the bufferSize queries RETURN calaman::Status, so a consumer
// importing this one module sees that type without a second import.
export import calaman.error_handling;
