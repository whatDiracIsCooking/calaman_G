# calaman.laqr1 -- LAPACK's ?laqr1, the first column of the shift polynomial
# (H - s1*I)(H - s2*I) for the multishift QR sweep. Pure host C++23: no BLAS and
# no device code -- ?laqr1 is O(1) scalar arithmetic on the top-left 3-by-3 of
# H, so it owns no .cu and touches only wwr.runtime_api to read H / write V on a
# stream.
#
# All links are PUBLIC because laqr1() is an exported template DEFINED in the
# interface (like calaman.lanv2): a consumer instantiates the body itself, so
# every module the body names must reach it. IMPORT_STD for std::abs.
# calaman.error_handling is additionally re-exported (export import) because it
# supplies the Status return type, whose member functions a consumer must see.
#   wwr.runtime_api         wwrStream_t + wwrMemcpy(Async) for H read/V write
#   calaman.error_handling  Status, the cross-domain return type
calaman_add_cxx_module_library(
  NAME
  calaman.laqr1
  PRIMARY_INTERFACE
  laqr1.cppm
  LINK_PUBLIC
  wwr.runtime_api
  calaman.error_handling
  IMPORT_STD
)
