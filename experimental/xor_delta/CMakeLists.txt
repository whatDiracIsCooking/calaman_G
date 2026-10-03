# calaman.experimental.xor_delta -- in-place XOR delta coding of a uint64_t
# stream. Pure host C++23: no BLAS, no device code, no WarpWraps link -- its
# only dependency is `import std` (IMPORT_STD), for std::uint64_t / std::size_t.
# It carries no LINK_* -- nothing in src/ reaches it and it reaches nothing.
calaman_add_cxx_module_library(
  NAME
  calaman.experimental.xor_delta
  PRIMARY_INTERFACE
  xor_delta.cppm
  IMPORT_STD
)
