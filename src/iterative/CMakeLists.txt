# calaman.iterative -- IterationInfo, the stop_reason concept and converged():
# the outcome report cg_unitary, feast and nnls share. Pure host, std only: no
# device code, no WarpWraps, so it links nothing.
calaman_add_cxx_module_library(
  NAME
  calaman.iterative
  PRIMARY_INTERFACE
  iterative.cppm
  IMPORT_STD
)
