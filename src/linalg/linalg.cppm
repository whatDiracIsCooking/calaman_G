/**
 * @file linalg.cppm
 * @brief Primary interface for calaman.linalg -- the dense linear-algebra
 *        routines, gathered from partitions
 *
 * There is ONE module under src/linalg, `calaman.linalg`, split into partitions
 * so each LAPACK-shaped routine keeps its own file -- the same assembly pattern
 * calaman.common uses. A routine that computes what a LAPACK routine computes
 * carries LAPACK's name (docs/architecture.md §4): `:larfg`, `:larf`, `:laqp2`,
 * `:geqp3`, `:laqps` land here as the geqp3 call graph is reconstructed
 * on-device from wrapped `wwr` BLAS plus a reflector toolkit.
 *
 * `export import :part` re-exports a partition's exported names to importers of
 * the module, so a consumer writes `import calaman.linalg;` once and sees every
 * `calaman::` name from every partition. A partition is NOT importable on its
 * own from outside the module -- that is the point: the partitions are internal
 * structure, and this unit is the single public face.
 *
 * This file is only the assembly point and starts empty: it exports nothing
 * until the first routine lands. Add a routine by adding its partition file
 * (src/linalg/<name>.cppm, `export module calaman.linalg:<name>;`) to the
 * PARTITIONS list in this directory's CMakeLists.txt and one `export import
 * :<name>;` below.
 *
 * Usage:
 *   import calaman.linalg;   // names every routine's exported symbols
 */

export module calaman.linalg;

// Partitions in call-graph order (larfg/larf, then laqp2, then geqp3/laqps).
export import :larfg;
