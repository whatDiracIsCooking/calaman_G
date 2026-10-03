# calaman.lanv2 -- LAPACK's ?lanv2, the Schur standardisation of a real 2-by-2
# nonsymmetric block. Pure host C++23: no BLAS and no device code -- ?lanv2 is
# scalar arithmetic on one 2-by-2, so it owns no .cu and no bridge header and
# touches only wwr.runtime_api to read/write the block on a stream.
#
# All links are PUBLIC because lanv2() is an exported template DEFINED in the
# interface (like calaman.larfg): a consumer instantiates the body itself, so
# every module the body names must reach it. IMPORT_STD for std::hypot /
# std::copysign / std::pow / numeric_limits. calaman.error_handling is
# additionally re-exported (export import) because it supplies the Status return
# type, whose member functions a consumer must see.
#   wwr.runtime_api         wwrStream_t + wwrMemcpy(Async) for the block r/w
#   calaman.error_handling  Status, the cross-domain return type
calaman_add_cxx_module_library(
  NAME
  calaman.lanv2
  PRIMARY_INTERFACE
  lanv2.cppm
  LINK_PUBLIC
  wwr.runtime_api
  calaman.error_handling
  IMPORT_STD
)
