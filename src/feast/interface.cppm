/**
 * @file interface.cppm
 * @brief Primary interface for calaman.feast
 *
 * FEAST eigensolver for real symmetric matrices: the eigenpairs with eigenvalues
 * in an interval. See driver.cppm and README.md. A shipped src/ module --
 * a host driver over wrapped batched BLAS (getrfBatched/getrsBatched), symm/gemm,
 * the wrapped eigensolver (syevd) and calaman.orthogonalize, plus the fused
 * device kernels of feast.cu -- written once against WarpWraps's `wwr*` names and
 * built for either backend.
 *
 * Exported: feast, feast_bufferSize, FeastOptions/FeastInfo/
 * FeastStopReason, feast_rational_filter, the feast_resolvent concept and its
 * optional feast_norm1_hook, and
 * linear_operator (calaman.linear_operator). The other partitions' declarations
 * are module-internal; they are re-exported only because every interface
 * partition must be.
 *
 * Usage:
 *   import calaman.feast;
 *   using namespace calaman;
 */

export module calaman.feast;

export import :quadrature;
export import :compute_quadrature;
export import :buffer_size;
export import :resolvent;
export import :rayleigh_ritz;
export import :driver;
