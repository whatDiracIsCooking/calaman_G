# calaman.syev -- LAPACK's ?syev, eigenvalues and optional eigenvectors of a
# real symmetric matrix. A host composition in calaman.geev's shape: lansy +
# lascl for the scaling, vendor ?sytrd / ?orgtr through wwr.wrappers.solver,
# then calaman.sterf or calaman.steqr (plus calaman.lacpy for the packed
# eigenvectors when lda > n). Pure host C++23, no .cu. Links are
# PUBLIC because syev() is an exported template defined in the interface.
calaman_add_cxx_module_library(
  NAME
  calaman.syev
  PRIMARY_INTERFACE
  syev.cppm
  LINK_PUBLIC
  wwr::blas
  wwr::solver
  wwr::runtime_api
  wwr::wrappers::solver
  calaman::common
  calaman::lansy
  calaman::lascl
  calaman::lacpy
  calaman::sterf
  calaman::steqr
  calaman::error_handling
  IMPORT_STD
)
