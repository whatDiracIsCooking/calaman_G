/**
 * @file interface.cppm
 * @brief Primary interface for calaman.davidson
 *
 * Block-Davidson eigensolver for the lowest n_roots eigenpairs of a symmetric
 * operator available only as a matrix-vector product. A shipped src/ module -- a
 * host driver over wrapped BLAS + the wrapped symmetric eigensolver (syevd, and
 * sygvd on the optional metric path) + caller-supplied operator/preconditioner
 * callbacks -- written once against WarpWraps's `wwr*` names and built for either
 * backend. Not a LAPACK routine (LAPACK ships no Davidson), so it is its own
 * module, like calaman.feast and calaman.expm.
 *
 * davidson_solve implements the EUCLIDEAN problem; the generalized (metric,
 * sygvd) path is still to come and a non-empty metric is rejected. See README.md.
 *
 * Usage:
 *   import calaman.davidson;
 *   using namespace calaman;
 */

export module calaman.davidson;

export import :buffer_size;
export import :solve;
