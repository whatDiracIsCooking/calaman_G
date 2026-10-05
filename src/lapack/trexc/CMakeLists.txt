# calaman.trexc -- LAPACK's ?trexc, reorder a real Schur factorization by moving
# a selected diagonal block to another position. A host composition over
# calaman.laexc: the block-reordering logic is sequential and index-dependent,
# so a host loop bubbles the moving block one adjacent swap (one laexc launch)
# at a time, reading back the subdiagonal entry that sizes a block and the
# device info laexc writes. The slim sibling of calaman.gehd2; owns no device
# code.
#
# All module links are PUBLIC because trexc() is an exported template DEFINED in
# the interface (like calaman.gehd2): a consumer instantiates the body, so every
# module the body names -- including calaman.laexc, the swap each step runs --
# must reach it. calaman.error_handling is additionally re-exported (export
# import) because it supplies the Status return type, whose member functions a
# consumer must see. trexc.cppm's GMF includes error_handling/error_macros.h
# (CLM_TRY), resolved root-relative via that target's exported include root.
#   wwr.runtime_api         wwrStream_t, wwrMemcpyAsync, wwrStreamSync, bad arg
#   calaman.common          real_fp -- the T constraint on the exported template
#   calaman.laexc           calaman::laexc -- the adjacent-block swap per step
#   calaman.error_handling  Status, the cross-domain return type (re-exported)
calaman_add_cxx_module_library(
  NAME
  calaman.trexc
  PRIMARY_INTERFACE
  trexc.cppm
  LINK_PUBLIC
  wwr.runtime_api
  calaman.common
  calaman.laexc
  calaman.error_handling
  IMPORT_STD
)
