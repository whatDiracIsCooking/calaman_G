# calaman.laqr0 -- LAPACK's ?laqr0, the small-bulge multishift QR driver with
# aggressive early deflation for the Schur form of a Hessenberg matrix. A host
# composition, the shape of calaman.laqr4 (which it reuses verbatim): the driver
# loop (active-block location, deflation-window sizing, shift selection / sort /
# shuffle) runs on the host and stages the O(1) scalars it inspects through
# device memcpies, while the heavy steps are the calaman.laqr3 (recursive AED),
# calaman.laqr4 (trailing-submatrix shifts), calaman.laqr5 (multishift sweep)
# and calaman.lahqr (tiny window) kernels. ?laqr0 IS ?laqr4 EXCEPT it uses the
# recursive ?laqr3 for AED (where ?laqr4 uses ?laqr2) and ?laqr4 for the
# trailing-submatrix shifts (where ?laqr4 uses ?lahqr) -- hence it links those
# shipped modules rather than inlining or recursing itself. Pure host C++23: no
# device code, no .cu, no bridge header.
#
# All module links are PUBLIC because laqr0() is an exported template DEFINED in
# the interface (like calaman.laqr4): a consumer instantiates the body, so every
# module the body names must reach it. calaman.error_handling is additionally
# re-exported (export import) because it supplies the Status return type, whose
# member functions a consumer must see; laqr0.cppm's GMF includes
# error_handling/error_macros.h (CLM_TRY), resolved root-relative via that
# target's exported include root.
#   wwr.runtime_api         wwrMemcpyAsync / wwrStreamSynchronize (staging)
#   wwr.wrappers.common     real_fp (the T constraint)
#   calaman.lahqr           tiny-window (n <= NTINY) Schur form
#   calaman.laqr3           the recursive aggressive early deflation
#   calaman.laqr4           the trailing-submatrix shift recomputation
#   calaman.laqr5           the multishift bulge-chasing sweep
#   calaman.error_handling  Status, the cross-domain return type (re-exported)
calaman_add_cxx_module_library(
  NAME
  calaman.laqr0
  PRIMARY_INTERFACE
  laqr0.cppm
  LINK_PUBLIC
  wwr.runtime_api
  wwr.wrappers.common
  calaman.lahqr
  calaman.laqr3
  calaman.laqr4
  calaman.laqr5
  calaman.error_handling
  IMPORT_STD
)
