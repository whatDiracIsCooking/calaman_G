# calaman.tri_index -- the header-only device index maps for the triangular
# storage formats (tri_index.cuh), shared by the storage-conversion kernels
# (tpttr, trttp, ...). Not a module: a kernel reaches the maps by #include.
#
# A consumer's device library links calaman::tri_index and includes
# "lapack/tri_index/tri_index.cuh". INTERFACE carries the src/ root that
# spelling resolves off, plus wwr.device for the <runtime.h> it includes.
calaman_add_interface_library(
  NAME
  calaman.tri_index
  LINK_PUBLIC
  wwr::device
)
target_include_directories(
  calaman.tri_index INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)
