# calaman.linear_operator -- linear_operator, the block-operator concept the
# matrix-free solvers share (feast's resolvent models; lanczos and davidson
# next, docs/architecture.md §9). Pure host, no device code. LINK_PUBLIC is
# what the interface imports: wwrStream_t (wwr.runtime_api) and Status
# (calaman.error_handling, re-exported).
calaman_add_cxx_module_library(
  NAME
  calaman.linear_operator
  PRIMARY_INTERFACE
  linear_operator.cppm
  LINK_PUBLIC
  wwr::runtime_api
  calaman::error_handling
  IMPORT_STD
)
