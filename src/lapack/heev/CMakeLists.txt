# calaman.heev -- LAPACK's ?heev, eigenvalues and optional eigenvectors of a
# complex Hermitian matrix. calaman.syev's shape over C/Z: lanhe + lascl for
# the scaling, vendor ?hetrd / ?ungtr through wwr.wrappers.solver, then
# calaman.sterf or complex calaman.steqr (plus calaman.lacpy for the packed
# eigenvectors when lda > n). Pure host C++23, no .cu. Links are PUBLIC
# because heev() is an exported template defined in the interface.
calaman_add_cxx_module_library(
  NAME
  calaman.heev
  PRIMARY_INTERFACE
  heev.cppm
  LINK_PUBLIC
  wwr::blas
  wwr::solver
  wwr::runtime_api
  wwr::complex
  wwr::wrappers::solver
  calaman::common
  calaman::lanhe
  calaman::lascl
  calaman::lacpy
  calaman::sterf
  calaman::steqr
  calaman::error_handling
  IMPORT_STD
)
