/**
 * @file laruv.cuh
 * @brief laruv_block: LAPACK's ?laruv, run cooperatively by one thread block
 *
 * The 48-bit multiplicative congruential generator of LAPACK 3.12.0's
 * slaruv/dlaruv, bit for bit: output i is the seed times multiplier row i of
 * kLaruvMultipliers, mod 2^48, held as four 12-bit limbs and scaled to (0,1).
 * Each output depends only on the incoming seed and its row, so thread i
 * computes output i; the reference's rare retry (an output that rounds to 1
 * bumps every seed limb by 2 for it and all later outputs) is reproduced by
 * re-running the tail from the first such index.
 *
 * calaman.laruv's kernel is one call of this; calaman.larnv's kernels call it
 * per chunk, with the mod-2^48 seed helpers below (laruv_join, laruv_powmod,
 * ...) to jump a seed ahead by whole calls. Reach it as "lapack/laruv/laruv.cuh" by linking the INTERFACE target
 * calaman::laruv::header. Device-only: include it from a .cu.
 *
 * Usage:
 *   #include "lapack/laruv/laruv.cuh"
 *   calaman::device::laruv_block(seed, n, x, iseed_out); // blockDim.x >= n
 */

#pragma once

#include "laruv_bridge.h"

#include <runtime.h>

namespace calaman::device {

/// @brief ?laruv's MM(128, 4): multiplier row i as four 12-bit limbs, high first
static __constant__ int kLaruvMultipliers[kLaruvMaxN][4] = {
    {494, 322, 2508, 2549}, {2637, 789, 3754, 1145}, {255, 1440, 1766, 2253},
    {2008, 752, 3572, 305}, {1253, 2859, 2893, 3301}, {3344, 123, 307, 1065},
    {4084, 1848, 1297, 3133}, {1739, 643, 3966, 2913}, {3143, 2405, 758, 3285},
    {3468, 2638, 2598, 1241}, {688, 2344, 3406, 1197}, {1657, 46, 2922, 3729},
    {1238, 3814, 1038, 2501}, {3166, 913, 2934, 1673}, {1292, 3649, 2091, 541},
    {3422, 339, 2451, 2753}, {1270, 3808, 1580, 949}, {2016, 822, 1958, 2361},
    {154, 2832, 2055, 1165}, {2862, 3078, 1507, 4081}, {697, 3633, 1078, 2725},
    {1706, 2970, 3273, 3305}, {491, 637, 17, 3069}, {931, 2249, 854, 3617},
    {1444, 2081, 2916, 3733}, {444, 4019, 3971, 409}, {3577, 1478, 2889, 2157},
    {3944, 242, 3831, 1361}, {2184, 481, 2621, 3973}, {1661, 2075, 1541, 1865},
    {3482, 4058, 893, 2525}, {657, 622, 736, 1409}, {3023, 3376, 3992, 3445},
    {3618, 812, 787, 3577}, {1267, 234, 2125, 77}, {1828, 641, 2364, 3761},
    {164, 4005, 2460, 2149}, {3798, 1122, 257, 1449}, {3087, 3135, 1574, 3005},
    {2400, 2640, 3912, 225}, {2870, 2302, 1216, 85}, {3876, 40, 3248, 3673},
    {1905, 1832, 3401, 3117}, {1593, 2247, 2124, 3089}, {1797, 2034, 2762, 1349},
    {1234, 2637, 149, 2057}, {3460, 1287, 2245, 413}, {328, 1691, 166, 65},
    {2861, 496, 466, 1845}, {1950, 1597, 4018, 697}, {617, 2394, 1399, 3085},
    {2070, 2584, 190, 3441}, {3331, 1843, 2879, 1573}, {769, 336, 153, 3689},
    {1558, 1472, 2320, 2941}, {2412, 2407, 18, 929}, {2800, 433, 712, 533},
    {189, 2096, 2159, 2841}, {287, 1761, 2318, 4077}, {2045, 2810, 2091, 721},
    {1227, 566, 3443, 2821}, {2838, 442, 1510, 2249}, {209, 41, 449, 2397},
    {2770, 1238, 1956, 2817}, {3654, 1086, 2201, 245}, {3993, 603, 3137, 1913},
    {192, 840, 3399, 1997}, {2253, 3168, 1321, 3121}, {3491, 1499, 2271, 997},
    {2889, 1084, 3667, 1833}, {2857, 3438, 2703, 2877}, {2094, 2408, 629, 1633},
    {1818, 1589, 2365, 981}, {688, 2391, 2431, 2009}, {1407, 288, 1113, 941},
    {634, 26, 3922, 2449}, {3231, 512, 2554, 197}, {815, 1456, 184, 2441},
    {3524, 171, 2099, 285}, {1914, 1677, 3228, 1473}, {516, 2657, 4012, 2741},
    {164, 2270, 1921, 3129}, {303, 2587, 3452, 909}, {2144, 2961, 3901, 2801},
    {3480, 1970, 572, 421}, {119, 1817, 3309, 4073}, {3357, 676, 3171, 2813},
    {837, 1410, 817, 2337}, {2826, 3723, 3039, 1429}, {2332, 2803, 1696, 1177},
    {2089, 3185, 1256, 1901}, {3780, 184, 3715, 81}, {1700, 663, 2077, 1669},
    {3712, 499, 3019, 2633}, {150, 3784, 1497, 2269}, {2000, 1631, 1101, 129},
    {3375, 1925, 717, 1141}, {1621, 3912, 51, 249}, {3090, 1398, 981, 3917},
    {3765, 1349, 1978, 2481}, {1149, 1441, 1813, 3941}, {3146, 2224, 3881, 2217},
    {33, 2411, 76, 2749}, {3082, 1907, 3846, 3041}, {2741, 3192, 3694, 1877},
    {359, 2786, 1682, 345}, {3316, 382, 124, 2861}, {1749, 37, 1660, 1809},
    {185, 759, 3997, 3141}, {2784, 2948, 479, 2825}, {2202, 1862, 1141, 157},
    {2199, 3802, 886, 2881}, {1364, 2423, 3514, 3637}, {1244, 2051, 1301, 1465},
    {2020, 2295, 3604, 2829}, {3160, 1332, 1888, 2161}, {2785, 1832, 1836, 3365},
    {2772, 2405, 1990, 361}, {1217, 3638, 2058, 2685}, {1822, 3661, 692, 3745},
    {1245, 327, 1194, 2325}, {2252, 3660, 20, 3609}, {3904, 716, 3285, 3821},
    {2774, 1842, 2046, 3537}, {997, 3987, 2107, 517}, {2573, 1368, 3508, 3017},
    {1148, 1848, 3525, 2141}, {545, 2366, 3801, 1537},
};

/// @brief One ?laruv draw: seed @p s times row @p i, mod 2^48, scaled to (0,1]
///
/// Writes the product's limbs to @p it (the reference's IT1..IT4). R = 2^-12
/// is a power of two, so every R*y below is exact and FMA contraction cannot
/// change the rounding.
template<typename T>
__device__ inline T laruv_draw(const int i, const int (&s)[4], int (&it)[4]) {
  constexpr int kIpw2 = 4096;
  constexpr T r = T(1) / T(kIpw2);
  const int *const mm = kLaruvMultipliers[i];
  int it4 = s[3] * mm[3];
  int it3 = it4 / kIpw2;
  it4 -= kIpw2 * it3;
  it3 += s[2] * mm[3] + s[3] * mm[2];
  int it2 = it3 / kIpw2;
  it3 -= kIpw2 * it2;
  it2 += s[1] * mm[3] + s[2] * mm[2] + s[3] * mm[1];
  int it1 = it2 / kIpw2;
  it2 -= kIpw2 * it1;
  it1 += s[0] * mm[3] + s[1] * mm[2] + s[2] * mm[1] + s[3] * mm[0];
  it1 %= kIpw2;
  it[0] = it1;
  it[1] = it2;
  it[2] = it3;
  it[3] = it4;
  return r * (T(it1) + r * (T(it2) + r * (T(it3) + r * T(it4))));
}

/// @brief ?laruv over the block: x[0:n) from @p seed; the new seed to @p seed_out
///
/// Every thread of the block must call it (it synchronizes), with the same
/// @p seed by value and blockDim.x >= @p n, 1 <= n <= kLaruvMaxN. Thread n-1
/// writes @p seed_out after the block's last read of any seed it was given.
/// Returns, in every thread, whether a draw rounded to 1 (the retry was taken).
template<typename T>
__device__ inline bool laruv_block(const int (&seed)[4], const int n, T *const x,
                                   int *const seed_out) {
  __shared__ int s_first_hit; // lowest index >= start whose draw rounded to 1
  const int i = static_cast<int>(threadIdx.x);
  int start = 0;
  int bump = 0; // the reference's cumulative +2 retries, applied to every limb
  T xi{};
  int it[4] = {};
  while (true) {
    if (i == 0) {
      s_first_hit = n;
    }
    __syncthreads();
    if (i >= start && i < n) {
      const int s[4] = {seed[0] + bump, seed[1] + bump, seed[2] + bump, seed[3] + bump};
      xi = laruv_draw<T>(i, s, it);
      if (xi == T(1)) {
        atomicMin(&s_first_hit, i);
      }
    }
    __syncthreads();
    const int first = s_first_hit;
    // Every thread reads s_first_hit before thread 0 resets it next pass.
    __syncthreads();
    if (first == n) {
      break;
    }
    start = first;
    bump += 2;
  }
  if (i < n) {
    x[i] = xi;
  }
  if (i == n - 1) {
    for (int k = 0; k < 4; ++k) {
      seed_out[k] = it[k];
    }
  }
  return bump > 0;
}

/// @brief The 48-bit value of four 12-bit limbs, high first (ISEED's layout)
__device__ __forceinline__ unsigned long long laruv_join(const int (&s)[4]) {
  return (static_cast<unsigned long long>(s[0]) << 36) |
         (static_cast<unsigned long long>(s[1]) << 24) |
         (static_cast<unsigned long long>(s[2]) << 12) | static_cast<unsigned long long>(s[3]);
}

/// @brief Split a 48-bit value into four 12-bit limbs, high first
__device__ __forceinline__ void laruv_split(const unsigned long long v, int (&s)[4]) {
  s[0] = static_cast<int>((v >> 36) & 4095);
  s[1] = static_cast<int>((v >> 24) & 4095);
  s[2] = static_cast<int>((v >> 12) & 4095);
  s[3] = static_cast<int>(v & 4095);
}

/// @brief a * b mod 2^48; the low 64 bits of the wrapped product suffice
__device__ __forceinline__ unsigned long long laruv_mulmod(const unsigned long long a,
                                                           const unsigned long long b) {
  return (a * b) & ((1ULL << 48) - 1);
}

/// @brief Multiplier row @p i as a 48-bit value
///
/// A ?laruv call of k draws with no retry leaves the seed times row k-1, so
/// whole calls compose by multiplication: the jump-ahead ?larnv chunks use.
__device__ __forceinline__ unsigned long long laruv_multiplier(const int i) {
  const int(&mm)[4] = kLaruvMultipliers[i];
  return laruv_join(mm);
}

/// @brief base^e mod 2^48 by binary exponentiation
__device__ inline unsigned long long laruv_powmod(unsigned long long base, unsigned long long e) {
  unsigned long long r = 1;
  while (e != 0) {
    if ((e & 1) != 0) {
      r = laruv_mulmod(r, base);
    }
    base = laruv_mulmod(base, base);
    e >>= 1;
  }
  return r;
}

} // namespace calaman::device
