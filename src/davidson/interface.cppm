/**
 * @file interface.cppm
 * @brief Primary interface for calaman.davidson
 *
 * Block-Davidson eigensolver for the lowest n_roots eigenpairs of a symmetric
 * operator available only as a linear_operator or a block matvec callback. A
 * shipped src/ module -- a host driver over wrapped BLAS + the wrapped symmetric
 * eigensolver (syevd, and sygvd on the optional metric path) + the caller's
 * operator and preconditioner -- written once against WarpWraps's `wwr*` names and built for either
 * backend. Not a LAPACK routine (LAPACK ships no Davidson), so it is its own
 * module, like calaman.feast and calaman.expm.
 *
 * davidson_solve handles both the standard (Euclidean, syevd) problem and the
 * generalized (metric, sygvd) one, selected by whether a metric callback is
 * supplied. See README.md.
 *
 * Usage:
 *   import calaman.davidson;
 *   using namespace calaman;
 */

export module calaman.davidson;

export import :buffer_size;
export import :solve;
