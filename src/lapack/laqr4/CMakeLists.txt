# calaman.laqr4 -- LAPACK's ?laqr4, the small-bulge multishift QR driver with
# aggressive early deflation. A host composition, the shape of calaman.gehrd:
# the driver loop (active-block location, deflation-window sizing, shift
# selection / sort / shuffle) runs on the host and stages the O(1) scalars it
# inspects through device memcpies, while the three heavy steps are the
# calaman.laqr2 (AED), calaman.lahqr (tiny window + trailing-submatrix shifts)
# and calaman.laqr5 (multishift sweep) kernels. ?laqr2 -- not the recursive
# ?laqr3 of ?laqr0 -- is what bounds the recursion, so this driver links those
# three shipped modules rather than inlining or recursing. Pure host C++23: no
# device code, no .cu, no bridge header.
#
# All module links are PUBLIC because laqr4() is an exported template DEFINED in
# the interface (like calaman.gehrd): a consumer instantiates the body, so every
# module the body names must reach it. calaman.error_handling is additionally
# re-exported (export import) because it supplies the Status return type, whose
# member functions a consumer must see; laqr4.cppm's GMF includes
# error_handling/error_macros.h (CLM_TRY), resolved root-relative via that
# target's exported include root.
#   wwr.runtime_api         wwrMemcpyAsync / wwrStreamSynchronize (staging)
#   wwr.wrappers.common     real_fp (the T constraint)
#   calaman.lahqr           tiny-window Schur form + trailing-submatrix shifts
#   calaman.laqr2           aggressive early deflation
#   calaman.laqr5           the multishift bulge-chasing sweep
#   calaman.error_handling  Status, the cross-domain return type (re-exported)
calaman_add_cxx_module_library(
  NAME
  calaman.laqr4
  PRIMARY_INTERFACE
  laqr4.cppm
  LINK_PUBLIC
  wwr.runtime_api
  wwr.wrappers.common
  calaman.lahqr
  calaman.laqr2
  calaman.laqr5
  calaman.error_handling
  IMPORT_STD
)
