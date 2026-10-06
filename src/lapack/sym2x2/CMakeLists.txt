# calaman.sym2x2 -- the header-only device ports of ?lae2, ?laev2 and ?lapy2
# (sym2x2.cuh), shared by every kernel that runs the symmetric tridiagonal chase
# (calaman.lae2/laev2/lapy2's oracle wrappers, sterf, steqr). Not a module: a
# kernel reaches the helpers by #include, not import.
#
# A consumer's device library links calaman::sym2x2 and includes
# "lapack/sym2x2/sym2x2.cuh". INTERFACE carries the src/ root that spelling
# resolves off, plus wwr.device for the <wrappers/math/math.cuh> it includes.
# The macro also exports this directory itself; spell the header root-relative
# anyway (codestyle). The install-interface spelling of the src/ root waits on
# the install tier (CALAMAN_INSTALL, docs/architecture.md §2).
calaman_add_interface_library(
  NAME
  calaman.sym2x2
  LINK_PUBLIC
  wwr::device
)
target_include_directories(
  calaman.sym2x2 INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)
