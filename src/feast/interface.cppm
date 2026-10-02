/**
 * @file interface.cppm
 * @brief Primary interface for calaman.feast
 *
 * FEAST eigensolver for real symmetric matrices: the eigenpairs with eigenvalues
 * in an interval. See feast_solver.cppm and README.md. A shipped src/ module --
 * a host driver over wrapped batched BLAS (getrfBatched/getrsBatched), symm/gemm,
 * the wrapped eigensolver (syevd) and calaman.orthogonalize, plus the fused
 * device kernels of feast.cu -- written once against WarpWraps's `wwr*` names and
 * built for either backend.
 *
 * Usage:
 *   import calaman.feast;
 *   using namespace calaman;
 */

export module calaman.feast;

export import :feast_quadrature;
export import :compute_quadrature;
export import :buffer_size;
export import :contour_filter;
export import :rayleigh_ritz;
export import :feast_solver;
