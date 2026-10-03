/**
 * @file interface.cppm
 * @brief Primary interface for calaman.cg_unitary
 *
 * Riemannian conjugate gradient for optimization under a unitary matrix
 * constraint, after Abrudan, Eriksson & Koivunen, "Conjugate gradient algorithm
 * for optimization under unitary matrix constraint", Signal Processing 89 (2009)
 * 1704-1714.
 *
 * Minimize or maximize a real-valued J(W) subject to W in U(n), by conjugate
 * gradient along geodesics of the unitary group, with the step size chosen by
 * one of the paper's two almost-periodicity line searches.
 *
 * Partitions:
 * - :cost_function    what the solver requires of J: order, gradient, value
 * - :brockett         J(W) = trace{W^H R W N}, the paper's §5.1 example
 * - :buffer_size      the single-buffer workspace layout and its sizing
 * - :geodesic_search  the line searches of Tables 1 and 2
 * - :cg_solver        the iteration of Table 3
 *
 * Usage:
 *   import calaman.cg_unitary;
 *   using namespace calaman;
 *
 *   std::size_t lwork = 0;
 *   brockett_cost<T> cost{d_R, n, d_N, n};
 *   cg_unitary_bufferSize<T>(cusolver, n, cost.bufferSize(n), &lwork);
 *   // ... allocate d_work, set W_0 = I ...
 *   cg_unitary<T>(cublas, cusolver, stream, n, d_W, cost,
 *                 CgDirection::Maximize, d_work, lwork);
 */

export module calaman.cg_unitary;

import std;

export import :cost_function;
export import :brockett;
export import :buffer_size;
export import :geodesic_search;
export import :cg_solver;

// cg_unitary() and cg_unitary_bufferSize() RETURN calaman::Status, so a consumer
// importing this one module sees that type without a second import -- the same
// re-export calaman.expm makes.
export import calaman.error_handling;
