/**
 * @file interface.cppm
 * @brief Primary interface for calaman.lanczos
 *
 * Thick-restart Lanczos for the extreme eigenpairs of a real symmetric operator
 * available only as a linear_operator or a single-vector matvec. A host driver over
 * wrapped BLAS + the wrapped symmetric eigensolver (syevd on the projected
 * matrix) + its own small device kernels, written once against WarpWraps's
 * `wwr*` names. Not a LAPACK routine, so it is its own module, like
 * calaman.davidson. :shift_invert reaches the interior: the pairs nearest a
 * shift, through an exact (A - sigma I)^{-1} operator. See README.md.
 *
 * Usage:
 *   import calaman.lanczos;
 *   using namespace calaman;
 */

export module calaman.lanczos;

export import :types;
export import :buffer_size;
export import :ritz;
export import :solve;
export import :shift_invert;
