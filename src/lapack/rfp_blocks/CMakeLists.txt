# calaman.rfp_blocks -- the host-side block geometry of Rectangular Full Packed
# storage: where A11, A22 and the off-diagonal block of an n-by-n triangle sit
# inside the RFP array (offset, leading dimension, stored uplo/transpose). Pure
# constexpr arithmetic shared by the routines that run BLAS on RFP sub-blocks
# (?hfrk/?sfrk, ?tfsm); the per-element map is calaman::tri_index's.
#
# calaman.common is PUBLIC: the exported surface names calaman::Uplo.
calaman_add_cxx_module_library(
  NAME
  calaman.rfp_blocks
  PRIMARY_INTERFACE
  rfp_blocks.cppm
  LINK_PUBLIC
  calaman::common
  IMPORT_STD
)
