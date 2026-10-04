# calaman.hseqr -- LAPACK's ?hseqr, the driver that computes the eigenvalues or
# the full real Schur form of an upper Hessenberg matrix. A host composition in
# the shape of calaman.laqr0: the driver logic (argument checks, the ?gebal
# eigenvalue isolation, Z initialisation, the n <= 1 quick returns, the trailing
# trash clear) runs on the host and stages the O(1) scalars it inspects through
# device memcpies, while the heavy QR iteration is one of two shipped kernels it
# dispatches to at the ILAENV(12) crossover NMIN = 75: calaman.laqr0 (multishift
# QR with recursive AED) for n > NMIN, else calaman.lahqr (double-shift Francis
# QR). calaman.laset supplies the Z-identity init and the strictly-lower trash
# clear. Pure host C++23: no device code, no .cu, no bridge header. Placed after
# laqr0 so both dispatch targets are already defined.
#
# All module links are PUBLIC because hseqr() is an exported template DEFINED in
# the interface (like calaman.laqr0): a consumer instantiates the body, so every
# module the body names -- lahqr, laqr0, laset -- must reach it.
# calaman.error_handling is additionally re-exported (export import) because it
# supplies the Status return type, whose member functions a consumer must see;
# hseqr.cppm's GMF includes error_handling/error_macros.h (CLM_TRY), resolved
# root-relative via that target's exported include root.
#   wwr.runtime_api         wwrMemcpyAsync / wwrStreamSynchronize (staging)
#   wwr.wrappers.common     real_fp (the T constraint)
#   calaman.lahqr           the n <= NMIN double-shift QR path
#   calaman.laqr0           the n > NMIN multishift-with-AED path
#   calaman.laset           Region + laset: Z identity init and trash clear
#   calaman.error_handling  Status, the cross-domain return type (re-exported)
calaman_add_cxx_module_library(
  NAME
  calaman.hseqr
  PRIMARY_INTERFACE
  hseqr.cppm
  LINK_PUBLIC
  wwr.runtime_api
  wwr.wrappers.common
  calaman.lahqr
  calaman.laqr0
  calaman.laset
  calaman.error_handling
  IMPORT_STD
)
