// geqp3_dump — dump the fp64 matrices the geqp3 test suite feeds to the device,
// at realistic sizes, so the compression spike measures REAL geqp3-path data
// instead of numpy stand-ins.
//
// The generator here is copied verbatim (in behaviour) from
// test/geqp3/geqp3_tests.cpp: a column-major matrix filled from
// std::mt19937(seed) + std::uniform_real_distribution<double>(-3, 3), optionally
// made rank-deficient by make_rank_deficient(rank) — columns [rank:n) become
// fixed linear combinations of the first `rank`, weights (c+1)/4. Compiled with
// the project's clang-20 + libc++, the std::mt19937 / uniform_real_distribution
// streams are bit-identical to the test's, so these are the same matrices the
// suite uses — only larger, since the test's own shapes (≤16) are too small for
// a meaningful ratio.
//
// Output: raw little-endian f64, column-major (lda = m), one file per case:
//   geqp3_<tag>_<m>x<n>[_r<rank>]_s<seed>.f64
//
//   clang++-20 -std=c++23 -stdlib=libc++ -O2 geqp3_dump.cpp -o geqp3_dump
//   ./geqp3_dump <out_dir>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

// Column-major fill, identical to expect_matches_reference's a_orig.
std::vector<double> make_matrix(int m, int n, unsigned seed) {
  const std::size_t lda = static_cast<std::size_t>(m);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-3.0, 3.0);
  std::vector<double> a(lda * static_cast<std::size_t>(n));
  for (auto& x : a) x = dist(rng);
  return a;
}

// Verbatim make_rank_deficient<double>: dependent columns are fixed linear
// combinations of the first `rank`, so the matrix has exactly `rank` independent
// columns.
void make_rank_deficient(std::vector<double>& a, int m, int n, int rank) {
  const std::size_t lda = static_cast<std::size_t>(m);
  for (int j = rank; j < n; ++j) {
    for (int i = 0; i < m; ++i) a[static_cast<std::size_t>(j) * lda + i] = 0.0;
    for (int c = 0; c < rank; ++c) {
      const double w = static_cast<double>(c + 1) / 4.0;
      for (int i = 0; i < m; ++i)
        a[static_cast<std::size_t>(j) * lda + i] +=
            w * a[static_cast<std::size_t>(c) * lda + i];
    }
  }
}

void dump(const std::filesystem::path& dir, const std::string& name,
          const std::vector<double>& a) {
  const auto path = dir / (name + ".f64");
  std::ofstream os(path, std::ios::binary);
  os.write(reinterpret_cast<const char*>(a.data()),
           static_cast<std::streamsize>(a.size() * sizeof(double)));
  std::printf("  %-34s %8zu values  %6.2f KiB\n", (name + ".f64").c_str(),
              a.size(), a.size() * sizeof(double) / 1024.0);
}

struct Case {
  const char* tag;
  int m, n, rank;  // rank <= 0 => full-rank dense
  unsigned seed;
};

}  // namespace

int main(int argc, char** argv) {
  const std::filesystem::path dir = argc > 1 ? argv[1] : "data";
  std::filesystem::create_directories(dir);

  // Shapes the geqp3 suite exercises (tall / wide / square, full-rank and
  // rank-deficient), scaled up to sizes where a compression ratio is meaningful.
  const Case cases[] = {
      {"dense_square", 512, 512, 0, 11},
      {"dense_tall", 512, 256, 0, 3},
      {"dense_wide", 256, 512, 0, 7},
      {"rankdef_r64", 512, 512, 64, 40},
      {"rankdef_r256", 512, 512, 256, 41},
      {"rankdef_wide_r96", 256, 512, 96, 42},
  };

  std::printf("dumping geqp3-path fp64 matrices to %s/\n", dir.c_str());
  for (const auto& c : cases) {
    auto a = make_matrix(c.m, c.n, c.seed);
    std::string name = std::string("geqp3_") + c.tag + "_" + std::to_string(c.m) +
                       "x" + std::to_string(c.n);
    if (c.rank > 0) {
      make_rank_deficient(a, c.m, c.n, c.rank);
      name += "_r" + std::to_string(c.rank);
    }
    name += "_s" + std::to_string(c.seed);
    dump(dir, name, a);
  }
  std::puts("done.");
  return 0;
}
