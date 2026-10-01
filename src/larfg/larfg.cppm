/**
 * @file larfg.cppm
 * @brief The calaman.larfg module -- generate an elementary Householder
 *        reflector, LAPACK's ?larfg
 *
 * Builds H = I - tau * v * v^T, with v = [1; v_tail], such that
 * H * [alpha; x] = [beta; 0]. Inputs are the leading scalar @p alpha and the
 * tail vector @p x (n-1 elements); outputs are beta (overwrites alpha), tau,
 * and the reflector tail v (overwrites x). This is the order-@p n reflector
 * LAPACK's ?larfg computes, the first routine in the geqp3 call graph.
 *
 * LAPACK's sign convention, reproduced exactly so the oracle agrees:
 *   beta = -sign(alpha) * sqrt(alpha^2 + ||x||^2)
 *   tau  = (beta - alpha) / beta
 *   v_tail = x / (alpha - beta)
 * When ||x|| == 0 the vector is already [alpha; 0]: H is the identity, so
 * tau = 0, beta = alpha, and the tail is left untouched (a zero tail is a
 * no-op reflector). The sign choice -sign(alpha) is what makes alpha - beta
 * the larger-magnitude difference, so the scaling never cancels catastrophically.
 *
 * No device code lives here. ||x|| is a stock nrm2 and the tail scaling is a
 * stock scal, so this reaches the OUTERMOST WarpWraps layer that does the job --
 * the type-safe dispatch wrappers in wwr.wrappers.blas (CLAUDE.md, "prefer the
 * outermost layer"). The scalar arithmetic (beta, tau, the reciprocal scale) is
 * three flops on three host-resident values; it is done on the host rather than
 * fused into a kernel, which keeps the routine a pure composition and backend-
 * neutral by construction. A future laqp2 calls this per column in a loop; the
 * one host<->device sync per call (reading alpha, writing it back) is acceptable
 * at that granularity.
 *
 * Requires the handle's DEFAULT (host) cuBLAS/hipBLAS pointer mode, like
 * calaman.diff_norm: nrm2 writes ||x|| to a host scalar and blocks, and scal's
 * reciprocal scale is passed as a HOST address, correct only in host pointer
 * mode. @p alpha and @p x are device pointers the caller owns; @p tau and @p
 * beta are host scalars this writes. alpha is read back once and, when scaling
 * happens, beta is written back to it so the device vector ends as
 * [beta; v_tail]. Those reads and writes go through the handle's OWN stream
 * (queried with wwrblasGetStream) so they order against both the caller's
 * uploads on that stream and this routine's nrm2/scal -- a bare default-stream
 * memcpy would race an upload the caller enqueued on a non-default stream.
 *
 * Templated over `float` and `double`. Complex ?larfg differs materially (a
 * complex tau, the conjugate in v^H, LAPACK's real-beta phase handling), so it
 * is a deliberate later extension, not a trivial instantiation -- the same wall
 * calaman.diff_norm and calaman.common/constants.h document.
 *
 * Usage:
 *   import calaman.larfg;
 *   import wwr.blas;   // wwrblasHandle_t, wwrblasCreate, WWRBLAS_STATUS_SUCCESS
 *   wwr::wwrblasHandle_t handle{};
 *   wwr::wwrblasCreate(&handle);
 *   // d_alpha: device scalar; d_x: device tail of length n-1
 *   double tau = 0.0, beta = 0.0;
 *   calaman::larfg(handle, n, d_alpha, d_x, 1, &tau, &beta);
 */

export module calaman.larfg;

import wwr.blas;               // wwrblasHandle_t, wwrblasStatus_t, WWRBLAS_STATUS_*
import wwr.runtime_api;        // wwrMemcpy(Async), wwrMemcpy{Device,Host}To*, wwrSuccess
import wwr.wrappers.blas;      // nrm2, scal
import std;                    // std::sqrt, std::abs

namespace calaman {

/// @brief Generate an elementary Householder reflector (LAPACK ?larfg)
///
/// Computes H = I - tau * v * v^T, v = [1; v_tail], with
/// H * [alpha; x] = [beta; 0]. Reads @p alpha and @p x from the device, forms
/// beta and tau on the host under LAPACK's sign convention, writes @p tau and
/// @p beta to the host scalars, and -- unless the tail is already zero --
/// overwrites @p alpha (device) with beta and scales @p x in place by
/// 1 / (alpha - beta) to become v_tail. A zero tail yields tau = 0, beta =
/// alpha, and leaves @p x untouched (the identity reflector).
///
/// Short-circuits: if nrm2 or the tail scal does not succeed its status is
/// returned and @p tau / @p beta are left unwritten. Returns success and writes
/// nothing when @p n <= 0. On the device-read/write failures that surface no
/// BLAS status, returns WWRBLAS_STATUS_NOT_INITIALIZED (the only neutral
/// non-success code WarpWraps exposes), matching calaman.diff_norm.
///
/// @tparam T Element type; one of the instantiated types (float, double)
/// @param handle GPU BLAS handle in host pointer mode; alpha and x live on its device
/// @param n Order of the reflector; the tail @p x has @p n - 1 elements
/// @param alpha Device scalar; the leading element, overwritten with beta
/// @param x Device tail of @p n - 1 elements, stride @p incx, overwritten with v_tail
/// @param incx Stride between elements of @p x
/// @param tau Host scalar; the reflector scalar tau is written here
/// @param beta Host scalar; the leading result beta is written here
/// @return The BLAS status of the failing step, otherwise WWRBLAS_STATUS_SUCCESS
export template<typename T>
wwr::wwrblasStatus_t larfg(wwr::wwrblasHandle_t handle, const int n, T *alpha, T *x, const int incx,
                           T *tau, T *beta) {
  if (n <= 0) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  // Order every scalar read/write on the handle's own stream, so they follow the
  // caller's uploads (enqueued on that same stream) and this routine's BLAS work.
  wwr::wwrStream_t stream{};
  if (wwr::wwrblasGetStream(handle, &stream) != wwr::WWRBLAS_STATUS_SUCCESS) {
    return wwr::WWRBLAS_STATUS_NOT_INITIALIZED;
  }

  // Read the leading scalar back and block until it lands; the tail has n - 1
  // elements, so n == 1 means an empty tail and a zero norm.
  T host_alpha{};
  if (wwr::wwrMemcpyAsync(&host_alpha, alpha, sizeof(T), wwr::wwrMemcpyDeviceToHost, stream) !=
          wwr::wwrSuccess ||
      wwr::wwrStreamSynchronize(stream) != wwr::wwrSuccess) {
    return wwr::WWRBLAS_STATUS_NOT_INITIALIZED;
  }

  const int tail = n - 1;
  T xnorm = T{0};
  if (tail > 0) {
    const auto status = wwr::nrm2<T>(handle, tail, x, incx, &xnorm);
    if (status != wwr::WWRBLAS_STATUS_SUCCESS) {
      return status;
    }
  }

  // Already [alpha; 0]: the identity reflector. tau = 0, beta = alpha, no scaling.
  if (xnorm == T{0}) {
    *tau = T{0};
    *beta = host_alpha;
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  // beta = -sign(alpha) * ||[alpha; x]||. sign(0) is taken as +1 so a zero alpha
  // with a nonzero tail gives a negative beta, as LAPACK's SIGN(alpha >= 0) does.
  const T r = std::sqrt(host_alpha * host_alpha + xnorm * xnorm);
  const T host_beta = host_alpha >= T{0} ? -r : r;
  *tau = (host_beta - host_alpha) / host_beta;
  *beta = host_beta;

  // v_tail = x / (alpha - beta); alpha - beta is the larger-magnitude difference
  // under this sign choice, so the scale is well conditioned.
  const T scale = T{1} / (host_alpha - host_beta);
  const auto status = wwr::scal<T>(handle, tail, &scale, x, incx);
  if (status != wwr::WWRBLAS_STATUS_SUCCESS) {
    return status;
  }

  // The device vector must end as [beta; v_tail], so write beta back over alpha
  // on the same stream, after the scal, and block until the write completes --
  // host_beta is a local, so it must not go out of scope before the copy runs.
  if (wwr::wwrMemcpyAsync(alpha, &host_beta, sizeof(T), wwr::wwrMemcpyHostToDevice, stream) !=
          wwr::wwrSuccess ||
      wwr::wwrStreamSynchronize(stream) != wwr::wwrSuccess) {
    return wwr::WWRBLAS_STATUS_NOT_INITIALIZED;
  }
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

} // namespace calaman
