/**
 * @file common.cppm
 * @brief Primary interface for calaman.common -- the shared, dependency-free
 *        building blocks, gathered from partitions
 *
 * There is ONE module under src/common, `calaman.common`, split into partitions
 * so each concern keeps its own file: `:constants` (constants.cppm, the typed
 * mathematical constants), `:align_up` (align_up.cppm, the integer rounding
 * helpers), `:enums` (enums.cppm, the LAPACK selector enums Jobz/Uplo),
 * `:workspace_builder` (workspace_builder.cppm, byte sizing for a device scratch
 * buffer) and `:validation` (validation.cppm, argument-checking predicates like
 * all_nonnull). This unit is only the assembly point.
 *
 * `export import :part` re-exports a partition's exported names to importers of
 * the module, so a consumer writes `import calaman.common;` once and sees every
 * `calaman::` name from every partition. A partition is NOT importable on its own
 * from outside -- `import calaman.common:constants;` is ill-formed in another
 * module -- which is the point: the partitions are internal structure, and this
 * is the single public face. Add a partition by adding one `export import` below.
 *
 * Usage:
 *   import calaman.common;
 *   const int blocks = calaman::idivup(n, threads_per_block);
 *   const double two_pi = calaman::kTwo<double> * calaman::kPi<double>;
 */

export module calaman.common;

export import :constants;
export import :align_up;
export import :enums;
export import :workspace_builder;
export import :validation;
