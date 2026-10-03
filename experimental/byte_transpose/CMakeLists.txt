# calaman.experimental.byte_transpose -- regroup an array of 64-bit values into
# eight planes by byte significance (block 0 = every value's MSB ... block 7 =
# its LSB), plus the inverse. Pure host C++23: no BLAS, no device code, no
# WarpWraps link -- its only dependency is `import std` (IMPORT_STD), for
# std::uint64_t / std::size_t / std::byte. It carries no LINK_* -- nothing in
# src/ reaches it and it reaches nothing.
calaman_add_cxx_module_library(
  NAME
  calaman.experimental.byte_transpose
  PRIMARY_INTERFACE
  byte_transpose.cppm
  IMPORT_STD
)
