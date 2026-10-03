/**
 * @file detail.cppm
 * @brief Shared internals for calaman.expm -- the complex-constant builder, the
 *        dimension bound, and the two device-workspace layouts
 *
 * The :detail partition of calaman.expm. Its names are `export`ed so the sibling
 * partitions (:pade, :buffer_size, :expm) can name them, but the primary
 * interface imports it WITHOUT re-exporting, so none of this reaches a consumer
 * of calaman.expm -- the same module-internal-only status these had as a plain
 * (non-export) namespace before the split.
 */

module;

#include "expm_bridge.h"

export module calaman.expm:detail;

import std;
import wwr.complex;         // wwrFloatComplex, wwrDoubleComplex, make_wwr*Complex (host)
import wwr.wrappers.common; // usual_fp, ComplexToRealType
import calaman.common;      // WorkspaceLayout

export namespace calaman {

// The fused kernels address elements with a flat int index, so n*n must fit in
// an int. floor(sqrt(INT_MAX)) = 46340; an n*n matrix that large is far past
// what a single device holds, but the bound is checked rather than assumed.
constexpr int kMaxDim = 46340;

/// @brief Build a real value as an element of T (complex types get a zero
///        imaginary part), the only complex constant the Pade evaluation needs.
///
/// Legal here, in the module purview, where wwr.complex is imported and its host
/// constructors are reachable -- unlike a GMF or a host constexpr, which cannot
/// see them (calaman.common's constants.h header documents the two walls).
template<wwr::usual_fp T>
T as_element(const wwr::ComplexToRealType<T> x) {
  if constexpr (std::is_same_v<T, wwr::wwrFloatComplex>) {
    return wwr::make_wwrFloatComplex(x, 0.0f);
  } else if constexpr (std::is_same_v<T, wwr::wwrDoubleComplex>) {
    return wwr::make_wwrDoubleComplex(x, 0.0);
  } else {
    return x;
  }
}

/**
 * @brief Device workspace slices for pade(), all 256-aligned.
 *
 * The layout is positional, and how many n*n blocks it spans depends on the
 * degree -- 3 at m = 3 and 6 at m = 9 and m = 13:
 *
 *   [P[0] | n*n T]  A^2 ... then holds U = A*W once V and W are formed
 *   [P[1] | n*n T]  A^4, when the degree needs it
 *   [P[2] | n*n T]  A^6,   "
 *   [P[3] | n*n T]  A^8 at m = 9; at m = 13 this slot is X, the nested inner
 *                   combination, used once per parity
 *   [V    | n*n T]  even part of the numerator ... then holds Q = V - U
 *   [W    | n*n T]  odd part, before multiplication by A
 *   [ipiv       | n int         ]
 *   [work_getrf | lwork_getrf T ]
 *
 * Two aliases keep the block count down. U reuses P[0]: the powers are all dead
 * by the time U is formed. Q reuses V: pade_split reads V(i) and writes Q(i)
 * from the same thread, so the denominator is built on top of the even half.
 */
template<wwr::usual_fp T>
struct PadeWorkspace {
  T *P[4] = {nullptr, nullptr, nullptr, nullptr};
  T *X = nullptr; // m = 13 only; aliases the P[3] slot
  T *V = nullptr;
  T *W = nullptr;
  T *U = nullptr; // aliases P[0] -- live only after V and W are formed
  T *Q = nullptr; // aliases V    -- live only after the split
  int *ipiv = nullptr;
  T *work_getrf = nullptr;

  /// @brief Carve the slices from @p layout, which sizes (null base) or carves
  ///        (real base) identically -- so pade_bufferSize and pade() share this
  ///        ONE region list and cannot drift. @p lwork_getrf is the getrf scratch
  ///        length, the one region whose size the caller has to query first.
  void carve(WorkspaceLayout &layout, const int n, const int m, const int lwork_getrf) {
    const std::size_t nn = static_cast<std::size_t>(n) * n;
    const int np = pade_num_powers(m);
    for (int k = 0; k < np; ++k) {
      P[k] = layout.fixed<T>(nn);
    }
    if (m == 13) {
      X = layout.fixed<T>(nn);
    }
    V = layout.fixed<T>(nn);
    W = layout.fixed<T>(nn);
    ipiv = layout.fixed<int>(static_cast<std::size_t>(n));
    work_getrf = layout.fixed<T>(static_cast<std::size_t>(lwork_getrf));
    U = P[0];
    Q = V;
  }
};

/**
 * @brief Device workspace slices for expm(), all 256-aligned.
 *
 *   [As     | n*n T                     ]  scaled matrix, and, once pade() has
 *                                          consumed it, the ping-pong target for
 *                                          the squaring phase -- hence the sq alias
 *   [colsum | n   ComplexToRealType<T>  ]  per-column absolute sums
 *   [pade   | PadeWorkspace, sized for the whole ladder ]
 */
template<wwr::usual_fp T>
struct ExpmWorkspace {
  using RealT = wwr::ComplexToRealType<T>;

  T *As = nullptr;
  T *sq = nullptr; // aliases As: only ever touched after pade() has returned
  RealT *colsum = nullptr;
  void *pade_base = nullptr;

  /// @brief Carve As and colsum from @p layout; pade_base is the layout cursor
  ///        after them -- where pade() carves its own sub-layout (expm sizes that
  ///        region for the worst degree, pade carves it for the chosen one).
  void carve(WorkspaceLayout &layout, const int n) {
    As = layout.fixed<T>(static_cast<std::size_t>(n) * n);
    sq = As;
    colsum = layout.fixed<RealT>(static_cast<std::size_t>(n));
    pade_base = layout.cursor();
  }
};

/**
 * @brief Device workspace slices for expm_herm(), all 256-aligned.
 *
 * The self-adjoint path exp(A) = U diag(exp(w)) U^H needs two n*n blocks and two
 * O(n) vectors -- far less than the Pade ladder:
 *
 *   [U        | n*n T          ]  the input A copied in, overwritten by the
 *                                 eigenvectors; also the right factor of the U^H
 *                                 product, so it must outlive the scaling
 *   [M        | n*n T          ]  U * diag(exp(w)), the left factor
 *   [w        | n RealT        ]  the (real) eigenvalues
 *   [eig_work | lwork_eig T    ]  the syevd/heevd scratch, whose length is the one
 *                                 region the caller must query first
 */
template<wwr::usual_fp T>
struct HermWorkspace {
  using RealT = wwr::ComplexToRealType<T>;

  T *U = nullptr;
  T *M = nullptr;
  RealT *w = nullptr;
  T *eig_work = nullptr;

  /// @brief Carve the slices from @p layout, which sizes (null base) or carves
  ///        (real base) identically -- so expm_herm_bufferSize and expm_herm
  ///        share this ONE region list and cannot drift. @p lwork_eig is the
  ///        eigensolver scratch length, the one region the caller queries first.
  void carve(WorkspaceLayout &layout, const int n, const int lwork_eig) {
    const std::size_t nn = static_cast<std::size_t>(n) * n;
    U = layout.fixed<T>(nn);
    M = layout.fixed<T>(nn);
    w = layout.fixed<RealT>(static_cast<std::size_t>(n));
    eig_work = layout.fixed<T>(static_cast<std::size_t>(lwork_eig));
  }
};

} // namespace calaman
